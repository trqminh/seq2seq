from utils import *
from model import *

vocabulary_size = len(string.ascii_lowercase) + 3  # [a-z] + ' ' + <sos> + <eos>
embedding_size = 400
hidden_size = 200
n_layers = 2
batch_size = 1
seq_len = 20
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.01

encoder_model = EncoderLSTM(vocab_size=vocabulary_size, embedding_size=embedding_size,
                            hidden_size=hidden_size, n_layers=n_layers).to(device)
decoder_model = DecoderLSTM(vocab_size=vocabulary_size, embedding_size=embedding_size,
                            hidden_size=hidden_size, output_size=vocabulary_size, n_layers=n_layers).to(device)

encoder_model.load_state_dict(torch.load('encoder.pth'))
decoder_model.load_state_dict(torch.load('decoder.pth'))

text = 'abc defg ijklm nop i'

encoder_inputs = list()
text_id = str2id(text)
text_id.append(2)
encoder_inputs.append(text_id)


encoder_inputs = torch.tensor(encoder_inputs).to(device)
encoder_hidden = encoder_model.init_hidden(batch_size=batch_size, device=device)
encoder_out, encoder_hidden = encoder_model(encoder_inputs, encoder_hidden)

decoder_inputs = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # <sos>
decoder_hidden = encoder_hidden

visual_answers = [[] for _ in range(batch_size)]
for di in range(seq_len+1):  # plus 1 for <eos>
    decoder_out, decoder_hidden = decoder_model(decoder_inputs, decoder_hidden)

    top_values, top_indices = decoder_out.topk(1)
    decoder_inputs = top_indices.squeeze(1).detach()

    for b in range(batch_size):
        visual_answers[b].append(decoder_inputs[b][:].item())

print(id2string(visual_answers[0]))







