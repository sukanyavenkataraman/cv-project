'''
Helper functions such as label_to_text, speller, decoder
'''
import re
import string
from collections import Counter
from keras import backend as K
import numpy as np

def text_to_labels(text):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret

def labels_to_text(labels):
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text

# Source: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

# Source: https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# Source: http://norvig.com/spell-correct.html (with some modifications)
class Spell(object):
    def __init__(self, path):
        self.dictionary = Counter(list(string.punctuation) + self.words(open(path).read()))

    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def P(self, word, N=None):
        "Probability of `word`."
        if N is None:
            N = sum(self.dictionary.values())
        return self.dictionary[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    # Correct words
    def corrections(self, words):
        return [self.correction(word) for word in words]

    # Correct sentence
    def sentence(self, sentence):
        return untokenize(self.corrections(tokenize(sentence)))

def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """Decodes the output of a softmax.
    Can use either greedy search (also known as best path)
    or a constrained dictionary search.
    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.
    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length,
                           greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    paths = [path.eval(session=K.get_session()) for path in decoded[0]]
    logprobs  = decoded[1].eval(session=K.get_session())

    return (paths, logprobs)

def decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, **kwargs):
    language_model = kwargs.get('language_model', None)

    paths, logprobs = _decode(y_pred=y_pred, input_length=input_length,
                              greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    if language_model is not None:
        # TODO: compute using language model
        raise NotImplementedError("Language model search is not implemented yet")
    else:
        # simply output highest probability sequence
        # paths has been sorted from the start
        result = paths[0]
    return result

class Decoder(object):
    def __init__(self, greedy=True, beam_width=100, top_paths=1, **kwargs):
        self.greedy         = greedy
        self.beam_width     = beam_width
        self.top_paths      = top_paths
        self.language_model = kwargs.get('language_model', None)
        self.postprocessors = kwargs.get('postprocessors', [])

    def decode(self, y_pred, input_length):
        decoded = decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width,
                         top_paths=self.top_paths, language_model=self.language_model)
        preprocessed = []
        for output in decoded:
            out = output
            for postprocessor in self.postprocessors:
                out = postprocessor(out)
            preprocessed.append(out)

        return preprocessed