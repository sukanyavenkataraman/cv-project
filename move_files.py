import os
from sklearn.cross_validation import train_test_split

print (os.listdir('.'))

x = y = os.listdir('.')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

for filen in x_train:
    os.rename(filen, '../../train_data_avletters/'+filen)

for filen in x_test:
    os.rename(filen, '../../valid_data_avletters/'+filen)

