import torch.nn as nn
import torch


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                            batch_first=True, num_layers=n_layers)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)

        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),\
               torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
