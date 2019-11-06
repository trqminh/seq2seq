# seq2seq
## description
in this repo, I have written and refactored some code to mirror string using seq2seq model
## references
+ https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py  
+ https://arxiv.org/abs/1409.3215  
+ https://github.com/vanhuyz/udacity-dl/blob/master/6_lstm-Problem3.ipynb  
## requirement
CUDA 9.0 + Pytorch 1.0
## run code
ROOT = path/to/this/repo
- download dataset
```python
cd ROOT
python dataset/download.py
```
- train model
```
python main.py
```
- inference:
```
python inference.py
```
## future works
attention, word embedding (glove or bert)
