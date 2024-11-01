import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x, length):
        #embeds = self.word_embeddings(sentence)
        #x = embeds.view(len(sentence), self.batch_size, -1)
        #x = x.view(self.batch_size, self.seq_len, -1)
        x_pack = rnn_utils.pack_padded_sequence(x, length, batch_first=False)
        lstm_out, self.hidden = self.lstm(x_pack, self.hidden)
        output2, out_len = rnn_utils.pad_packed_sequence(lstm_out, batch_first=False)
        tmp_hidden = [output2[length[bid]-1][bid] for bid in range(len(length))]
        last_hidden = torch.stack(tmp_hidden)
        y  = self.hidden2label(last_hidden)
        return y

