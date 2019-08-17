from utils import *
from model import *
import torch.optim as optim


def train(batch_gen, n_iter):

    vocabulary_size = len(string.ascii_lowercase) + 3  # [a-z] + ' ' + <sos> + <eos>
    embedding_size = 400
    hidden_size = 200
    n_layers = 2
    batch_size = 4
    seq_len = batch_gen.get_seq_len()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.001

    criterion = nn.NLLLoss()

    encoder_model = EncoderLSTM(vocab_size=vocabulary_size, embedding_size=embedding_size,
                                hidden_size=hidden_size, n_layers=n_layers).to(device)
    encoder_optim = optim.SGD(encoder_model.parameters(), lr=learning_rate)
    decoder_model = DecoderLSTM(vocab_size=vocabulary_size, embedding_size=embedding_size,
                                hidden_size=hidden_size, output_size=vocabulary_size, n_layers=n_layers).to(device)
    decoder_optim = optim.SGD(decoder_model.parameters(), lr=learning_rate)

    for it in range(n_iter):
        loss = 0

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        batch = batch_gen.next()

        # ---------- ENCODE PHASE ----------
        encoder_inputs = []
        for seq in batch:
            encoder_inputs.append(str2id(seq))

        encoder_inputs = torch.tensor(encoder_inputs).to(device)

        encoder_hidden = encoder_model.init_hidden(batch_size=batch_size, device=device)
        encoder_out, encoder_hidden = encoder_model(encoder_inputs, encoder_hidden)

        # ---------- DECODE PHASE ----------

        targets = mirror_batches(batch)
        target_tensor = []
        for target in targets:
            target_tensor.append(str2id(target))

        target_tensor = torch.tensor(target_tensor).to(device)

        decoder_inputs = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # <sos>
        decoder_hidden = encoder_hidden

        eval_out = []
        for di in range(seq_len):
            decoder_out, decoder_hidden = decoder_model(decoder_inputs, decoder_hidden)

            top_values, top_indices = decoder_out.topk(1)
            decoder_inputs = top_indices.squeeze(1).detach()

            loss += criterion(decoder_out.squeeze(1), target_tensor[:, di])

        loss.backward()
        encoder_optim.step()
        decoder_optim.step()

        if it % 500 == 0:
            print('-----------------')
            print('iter ', it)
            print('loss: ', loss.item())
            print(batch[0])




