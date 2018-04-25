''''
Generate the avletter dictionary similar to grid
'''

num_each_letter = 30

with open('avletters.txt', 'w+') as f:
    for i in range(26):
        for j in range(num_each_letter):
            letter = chr(ord('a')+i)
            f.writelines(letter+'\n')

f.close()