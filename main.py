import os
import torch
import copy
from torch.utils.data import DataLoader
import dataset as DP
import sequence_generate as sg
from lstm_model import LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

use_seq_generate_predict = False

use_plot = True
use_save = True
if use_save:
    import pickle
    from datetime import datetime

DATA_DIR = 'data'
TRAIN_FILE = 'arrange_all.xlsx'


## parameter setting
epochs = 100
batch_size = 5
use_gpu = torch.cuda.is_available()
learning_rate = 0.05


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_x = [i[0] for i in data]
    label = [i[1] for i in data]

    data_length = [len(sq) for sq in data_x]
    data_x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0)
    return data_x, label, data_length


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    lr=learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__=='__main__':
    ### parameter setting
    input_dim = 2
    hidden_dim = 30
    nlabel = 6
    sheet="mp"
    #sentence_len = 32
    total_file_path = os.path.join(DATA_DIR, TRAIN_FILE)

    ### create model
    model = LSTMC(input_dim=input_dim,hidden_dim=hidden_dim,
                           label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    ### data processing
    if use_seq_generate_predict:
        total_set = sg.SeqGenerator(DATA_DIR, TRAIN_FILE, sheet)
    else:
        total_set = DP.TxtDataset(DATA_DIR,TRAIN_FILE,sheet)


    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_size = int(len(total_set) * train_ratio)
    val_size = int(len(total_set) * val_ratio)
    test_size = len(total_set) - val_size - train_size

    dtrain_set, dval_set, dtest_set = torch.utils.data.random_split(total_set, [train_size, val_size, test_size])
    train_loader = DataLoader(dtrain_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn
                         )

    val_loader = DataLoader(dval_set,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=collate_fn
                         )

    test_loader = DataLoader(dtest_set,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=collate_fn
                         )

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    target = torch.tensor([1, -1])
    seq_generator_loss_function = nn.CosineEmbeddingLoss()
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    ### training procedure
    for epoch in range(epochs):
        optimizer = adjust_learning_rate(optimizer, epoch)

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader):
            train_inputs, train_labels, length = traindata
            train_labels = torch.squeeze(torch.tensor(train_labels))
            if use_gpu:
                train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()
            else: train_inputs = train_inputs

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()


            output = model(train_inputs.transpose(0,1),length)


            #loss = loss_function(output, train_labels)
            if use_seq_generate_predict:
                loss = seq_generator_loss_function(output, train_labels,target)
            else:
                loss = loss_function(output, train_labels)
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)
        ## testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, valdata in enumerate(val_loader):
            val_inputs, val_labels, val_length = valdata
            val_labels = torch.squeeze(torch.tensor(val_labels))
            if val_labels.dim() == 0:
                val_labels = val_labels.unsqueeze(0)
            if use_gpu:
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
            else: val_inputs = val_inputs

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs.transpose(0,1),val_length)

            loss = loss_function(output, val_labels)

            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, epochs, train_loss_[epoch], val_loss_[epoch], train_acc_[epoch], val_acc_[epoch]))

    param = {}
    param['lr'] = learning_rate
    param['batch size'] = batch_size
    param['input dim'] = input_dim
    param['hidden dim'] = hidden_dim

    result = {}
    result['train loss'] = train_loss_
    result['val loss'] = val_loss_
    result['train acc'] = [acc.cpu() for acc in train_acc_]
    result['val acc'] = [acc.cpu() for acc in val_acc_]
    result['param'] = param

    # if use_plot:
    #     import PlotFigure as PF
    #     PF.PlotFigure(result, use_save)
    if use_save:
        filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['filename'] = filename

        fp = open(filename, 'wb')
        pickle.dump(result, fp)
        fp.close()
        print('File %s is saved.' % filename)
