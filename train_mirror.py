from utils import *
from model import *
import torch.optim as optim
import copy


def train(train_batch_gen, val_batch_gen, n_iter):

    vocabulary_size = len(string.ascii_lowercase) + 3  # [a-z] + ' ' + <sos> + <eos>
    embedding_size = 400
    hidden_size = 200
    n_layers = 2
    batch_size = 4
    seq_len = train_batch_gen.get_seq_len()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.01

    criterion = nn.CrossEntropyLoss()

    encoder_model = EncoderLSTM(vocab_size=vocabulary_size, embedding_size=embedding_size,
                                hidden_size=hidden_size, n_layers=n_layers).to(device)
    encoder_optim = optim.SGD(encoder_model.parameters(), lr=learning_rate)
    decoder_model = DecoderLSTM(vocab_size=vocabulary_size, embedding_size=embedding_size,
                                hidden_size=hidden_size, output_size=vocabulary_size, n_layers=n_layers).to(device)
    decoder_optim = optim.SGD(decoder_model.parameters(), lr=learning_rate)

    best_encoder_model = copy.deepcopy(encoder_model.state_dict())
    best_decoder_model = copy.deepcopy(decoder_model.state_dict())
    best_acc = 0.0

    for it in range(n_iter):
        if it % 500 == 0:
            print('##############')
            print('iter ', it)

        for phase in ['train', 'val']:

            encoder_model.train()
            decoder_model.train()
            batch = train_batch_gen.next()

            if phase == 'val':
                encoder_model.eval()
                decoder_model.eval()
                batch = val_batch_gen.next()

            loss = 0

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()

            # ---------- ENCODE PHASE ----------
            encoder_inputs = []
            for seq in batch:
                encoder_inputs.append(str2id(seq))

            for b in range(batch_size):
                encoder_inputs[b].append(2)  # append <eos>

            encoder_inputs = torch.tensor(encoder_inputs).to(device)
            encoder_hidden = encoder_model.init_hidden(batch_size=batch_size, device=device)
            encoder_out, encoder_hidden = encoder_model(encoder_inputs, encoder_hidden)

            # ---------- DECODE PHASE ----------

            targets = mirror_batches(batch)
            target_tensor = []
            for target in targets:
                target_tensor.append(str2id(target))

            for b in range(batch_size):
                target_tensor[b].append(2)  # append <eos>

            target_tensor = torch.tensor(target_tensor).to(device)

            decoder_inputs = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # <sos>
            decoder_hidden = encoder_hidden

            visual_answers = [[] for _ in range(batch_size)]
            count_true = 0
            for di in range(seq_len+1):  # plus 1 for <eos>
                decoder_out, decoder_hidden = decoder_model(decoder_inputs, decoder_hidden)

                top_values, top_indices = decoder_out.topk(1)
                decoder_inputs = top_indices.squeeze(1).detach()

                for b in range(batch_size):
                    visual_answers[b].append(decoder_inputs[b][:].item())

                loss += criterion(decoder_out.squeeze(1), target_tensor[:, di])
                count_true += ((decoder_inputs.view(-1) == target_tensor[:, di]).sum().item())

            acc = count_true / (batch_size * (seq_len+1))

            if phase == 'train':
                loss.backward()
                encoder_optim.step()
                decoder_optim.step()

            if phase == 'val':
                if acc > best_acc:
                    best_acc = acc
                    best_encoder_model = copy.deepcopy(encoder_model.state_dict())
                    best_decoder_model = copy.deepcopy(decoder_model.state_dict())

            if it % 500 == 0:
                if phase == 'train':
                    print('training loss: ', loss.item() / seq_len)
                    print(batch[0])
                    print(id2string(visual_answers[0]))
                else:
                    print('-----------------')
                    print('valid loss: ', loss.item() / seq_len)
                    print('acc: ', acc)
                    print(batch[0])
                    print(id2string(visual_answers[0]))

    print('best acc: ', best_acc)
    torch.save(best_encoder_model, 'encoder.pth')
    torch.save(best_decoder_model, 'decoder.pth')
    return best_encoder_model, best_decoder_model
