from train_mirror import *

text = read_data('./dataset/text8.zip')

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

train_batch_gen = BatchGenerator(train_text, 4, 20)
val_batch_gen = BatchGenerator(valid_text, 4, 20)


train(train_batch_gen, val_batch_gen, 50001)
