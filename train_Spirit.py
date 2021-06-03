from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np

from utils import *
from modules import *
import os

from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Porn',
                    help='Type of  model.')
parser.add_argument('--dataset', type=str, default='Opportunity',
                    help='Dataset we used(Realdisp)')

parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.(0.0005)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--l2', type=float, default=1e-5,
                    help='L2 Reguralization')
parser.add_argument('--Scheduling_lambda', type=float, default=0.996,
                    help='Scheduling lambda')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Number of samples per batch.')

parser.add_argument('--ex-num', type=int, default=0, 
                    help='ID of experiments.')
parser.add_argument('--test-user', type=int, default=0, 
                    help='ID of test user.')
parser.add_argument('--num-atoms', type=int, default=5, 
                    help='Number of atoms in simulation.')
parser.add_argument('--dims', type=int, default=9,
                    help='The number of input dimensions.')
parser.add_argument('--timesteps', type=int, default=300,
                    help='The number of time steps per sample.')


parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed) 
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms) # (7, 7)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)   # size = (42, 7)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # size = (42, 7)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)    
# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms) # 上三角元素index
tril_indices = get_tril_offdiag_indices(args.num_atoms) # 下三角元素index
if args.cuda:
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()
rel_rec = Variable(rel_rec)      # torch.Size([42, 7])
rel_send = Variable(rel_send)    # torch.Size([42, 7])


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []

    model.train()
    
    criterion = nn.CrossEntropyLoss().cuda()
    for batch_idx, (data, relations) in enumerate(train_loader):
    # (32, 5, 300, 9), (32)
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)
        optimizer.zero_grad()
        
        output = model(data, rel_rec, rel_send) # (32, 5)
            
        loss = criterion(output, relations) 

        loss.backward()
        optimizer.step()

        loss_train.append(loss.data.item())
        
    scheduler.step()          


    loss_val = []

    model.eval()
    correct1 = 0
    size = 0    
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        output = model(data, rel_rec, rel_send)

        test_loss = criterion(output, relations)
        
        pred1 = output.data.max(1)[1] 
        k = relations.data.size()[0]
        correct1 += pred1.eq(relations.data).cpu().sum()
        size += k        
        
        loss_val.append(test_loss.data.item())
        
    writer.add_scalar('loss_train', np.mean(loss_train), global_step=epoch)
    writer.add_scalar('loss_val', np.mean(loss_val), global_step=epoch)
    writer.add_scalar('test_acc', 1. * correct1.float() / size, global_step=epoch)

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.10f}'.format(np.mean(loss_train)),
          'loss_val: {:.10f}'.format(np.mean(loss_val)),
          'test_acc:{:.6f}'.format(1. * correct1.float() / size),
          'time: {:.4f}s'.format(time.time() - t))
    return np.mean(loss_val)  
    
# Train model
for i in range(4):
    if args.model == 'Porn':
        model = Porn()
    elif args.model == 'Wdk1_':
        model = Mom4_()       
    model.cuda()    

    optimizer = optim.Adam(list(model.parameters()),lr=args.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)  
    
    lambda1 = lambda epoch: args.Scheduling_lambda ** epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    args.test_user = i
    save_path =  args.model + str(args.ex_num) + '/' + 'user' + str(args.test_user) + '/'
    writer = SummaryWriter(save_path)
    train_loader, test_loader = load_data(args.batch_size, test_user= args.test_user)

    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss)
        