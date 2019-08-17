import torch.nn as nn
import torch
import torch.nn.functional as F


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, n_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                            batch_first=True, num_layers=n_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, decoder_inputs, hidden):
        embeds = self.embedding(decoder_inputs)
        out = F.relu(embeds)

        out, hidden = self.lstm(out, hidden)
        out = self.fc(out)
        out = self.softmax(out)

        return out, hidden

