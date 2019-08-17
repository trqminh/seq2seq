from utils import *
from train_mirror import *
import torch

text = read_data('./dataset/text8.zip')

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

batch_gen = BatchGenerator(train_text, 4, 20)

train(batch_gen, 10000)




