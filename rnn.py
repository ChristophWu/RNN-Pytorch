# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:59:50 2018

@author: Jason
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data

# Load data and proprocessing
data_URL = 'shakespeare_train.txt'
with open(data_URL, 'r') as f:
    text = f.read()

# Characters' collection
vocab = set(text)

# Construct character dictionary
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

# Encode data, shape = [# of characters]
train_encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

vallidation_data_url = 'shakespeare_valid.txt'
with open(vallidation_data_url, 'r') as f:
    text = f.read()
valid_encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

# # how many sequences in one batch
# batch_size = 10
# num_steps = 50
# # characters in one batch
# BATCH_SIZE = batch_size * num_steps
# train_loader = Data.DataLoader(dataset=train_encode, batch_size=BATCH_SIZE, shuffle=False)


# Divide data into mini-batches
# -------------------------------------------------------------#
def get_batches(arr, n_seqs, n_steps):
    
    '''
    arr: data to be divided
    n_seqs: batch-size, # of input sequences
    n_steps: timestep, # of characters in a input sequences
    '''
    
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

# Function above define a generator, call next() to get one mini-batch
batch_size = 80
num_steps = 50
train_batches = get_batches(train_encode, batch_size, num_steps)
x, y = next(train_batches)

# model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(  # 这回一个普通的 RNN 就能胜任
            input_size=68,
            hidden_size=256,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(256, 68)

    def forward(self, x, h_state):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        r_out, h_state = self.rnn(x, h_state)
        outs = []    # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):    # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


rnn = RNN()
# if gpu is available
# rnn.cuda()
print(rnn)

# training
LR = 0.1
EPOCH = 20
num_labels = 68
optimizer = torch.optim.SGD(rnn.parameters(), lr=LR)   # optimize all rnn parameters
loss_func = nn.CrossEntropyLoss(size_average=False)
train_loss = []
valid_loss = []
for epoch in range(EPOCH):
    h_state = None
    k = 0
       # 要使用初始 hidden state, 可以设成 None
    for x, y in get_batches(train_encode, batch_size, num_steps):
        x = np.array(x)
        x_onehot = x.reshape([-1])
        x_onehot = (np.arange(num_labels) == x_onehot[:,None]).astype(np.float32).reshape([batch_size,num_steps,num_labels])
        x = Variable(torch.from_numpy(x_onehot) )    # shape (batch, time_step, input_size)
        y = Variable(torch.from_numpy(y[:,:]).type(torch.LongTensor))
#       #for lstm
#         prediction, (h_state_c,h_state_h) = rnn(x, h_state)  # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
#         # !!  下一步十分重要 !!
#         h_state_c = Variable(h_state_c.data)  # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错
#         h_state_h = Variable(h_state_h.data)
#         h_state = (h_state_c,h_state_h)
        prediction, h_state = rnn(x, h_state)
        h_state = Variable(h_state.data)
        prediction = prediction.view(-1,68)
        y = y.view(-1)
        loss = loss_func(prediction, y)/ (batch_size * num_steps)     # cross entropy loss
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        if (k % 100 == 0):
            print('epoch:',epoch,'loss:',loss.data.numpy())
        k = k + 1
        
    print('epoch:',epoch,'training_loss',loss.data.numpy())
    train_loss.append(loss.data.numpy())
    k = 0
    tmp = []
    for x, y in get_batches(valid_encode, batch_size, num_steps):
        x = np.array(x)
        x_onehot = x.reshape([-1])
        x_onehot = (np.arange(num_labels) == x_onehot[:,None]).astype(np.float32).reshape([batch_size,num_steps,num_labels])
        x = Variable(torch.from_numpy(x_onehot) )    # shape (batch, time_step, input_size)
        y = Variable(torch.from_numpy(y[:,:]).type(torch.LongTensor))
        prediction, h_state = rnn(x, h_state)
        prediction = prediction.view(-1,68)
        y = y.view(-1)
        loss = loss_func(prediction, y)/ (batch_size * num_steps) 
        tmp.append(loss.data.numpy())
        k = k + 1
    tmp = np.array(tmp)
    loss_tmp = sum(tmp)/k
    print('epoch:',epoch,'valid_loss',loss_tmp)
    valid_loss.append(loss_tmp)
    
# draw the learning curve
import matplotlib.pyplot as plt
t = np.arange(20)
train_loss_ = np.array(train_loss).reshape(-1)
valid_loss_ = np.array(valid_loss).reshape(-1)
print(train_loss_)
p1 = plt.plot(t,train_loss_,'r', label='training_loss')
p2 = plt.plot(t,valid_loss_,'b', label='test_loss')
plt.title('training and test loss')
# plt.title('learning curve')
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.legend(loc='upper right') 
plt.show()

