# from udacity deep learning course

import zipfile
import tensorflow as tf
import string

def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


def char2id(char):
    first_letter = ord(string.ascii_lowercase[0])

    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 3
    elif char == ' ':
        return 0
    elif char == '<sos>':
        return 1
    elif char == '<eos>':
        return 2
    else:
        print('Unexpected character: %s' % char)
        return


def id2char(dict_id):
    first_letter = ord(string.ascii_lowercase[0])

    if dict_id > 2:
        return chr(dict_id + first_letter - 3)
    elif dict_id == 1:
        return '<sos>'
    elif dict_id == 2:
        return '<eos>'
    else:
        return ' '


def str2id(s):
    return [char2id(c) for c in s]


def id2string(list_c):
    s = ''
    for c in list_c:
        s += id2char(c)

    return s


def reverse_word(word):
    return word[::-1]


def mirror_sequence(seq):
    return ' '.join(list(map(lambda word: reverse_word(word), seq.split(' '))))


def mirror_batches(batches):
    return [mirror_sequence(seq) for seq in batches]
