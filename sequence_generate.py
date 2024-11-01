import os
import torch
import copy
from torch.utils.data import DataLoader
import dataset as DP
from lstm_model import LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data.dataset import Dataset
from dataset import deal_excel_file
from main import collate_fn


class SeqGenerator(Dataset):
    def __init__(self,data_path,excel_path, sheet_name):
        self.excel_path = os.path.join(data_path, excel_path)
        self.main_x, self.label, self.attr_names = deal_excel_file(self.excel_path, sheet_name)
        ngram_seq = ngram_generate(self.main_x)
        self.x, self.y = pad_seq_generate(ngram_seq)
        print(len(self.x), len(self.y))
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def ngram_generate(data):
    ngram_sequences=[]
    for vec in data:
        if len(vec) < 2:
            continue
        seq_len = len(vec)
        for i in range(seq_len-1):
            ngram_sequences.append(vec[:i+2])
    return ngram_sequences

def pad_seq_generate(seqs):
    max_seq_len = max([len(seq) for seq in seqs])
    input_seqs = [torch.tensor(i) for i in seqs]
    x = [i[:-1] for i in input_seqs]
    y = [i[-1] for i in input_seqs]
    #pad_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    #length = torch.tensor([len(i) for i in x])
    return x,y

if __name__=="__main__":
    DATA_DIR = 'data'
    TRAIN_FILE = 'arrange_all.xlsx'
    data=[[1,2,3],[4,5,6],[2,3,1,43,21,421],[12,49,21,60],[1,2,5]]
    #ngram_seq=ngram_generate(data)
    #x,y,length=pad_seq_generate(ngram_seq,5)
    seq_set=SeqGenerator(DATA_DIR,TRAIN_FILE,'protein')
    train_loader = DataLoader(seq_set,
                              batch_size=5,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn
                              )
    for iter, traindata in enumerate(train_loader):
        print(train_data)