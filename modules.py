import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax
from modelsummary import summary

_EPS = 1e-10

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):   
        return  F.relu(self.batch_norm(self.fc1(inputs)))

class Porn (nn.Module):
    """Porn_ decoder module."""
    def __init__(self):
        super(Porn , self).__init__()
        self.dropout_prob = 0.5
        self.conv1 = nn.Conv2d(1, 64, (5, 9))
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, 
                         padding=0,dilation=1, return_indices=False, ceil_mode=False)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, num_layers=1)  
        
        self.conv6 = nn.Conv2d(1, 64, (1, 1))
        self.conv7 = nn.Conv2d(64, 64, (5, 1))       
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)        
        
        self.Q = nn.Conv2d(64, 1, (1, 1))
        self.K = nn.Conv2d(64, 1, (1, 1))
        self.V = nn.Conv2d(64, 1, (1, 1))
        self.bnQ = nn.BatchNorm2d(1)
        self.bnK = nn.BatchNorm2d(1)
        self.bnV = nn.BatchNorm2d(1)
            
        self.out = nn.Linear(64, 2)
        self.fc_out = nn.Linear(512, 5)
        
        self.mlp1 = MLP(256, 256, 256, 0.5)
        self.mlp2 = MLP(256, 256, 256, 0.5)
        self.mlp3 = MLP(768, 256, 64, 0.5)
        
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(2)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(2)])  
        self.msg_fc3 = nn.Linear(512, 64)
        
        print("This is Porn ")
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)        # size = [5,20]*[32,20,256]
        return incoming / incoming.size(1)             # size = [32,5,256]

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)            # size = [20, 5]*[32,5,128] => [32,20,128]
        senders = torch.matmul(rel_send, x)             # size = [20, 5]*[32,5,128] => [32,20,128]
        edges = torch.cat([senders, receivers], dim=2)  # size = [32, 20, 256]
        return edges        

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):
        receivers = torch.matmul(rel_rec, single_timestep_inputs) # [20, 5]*[32, 1, 5, 128],  (32, 1, 20, 128)
        senders = torch.matmul(rel_send, single_timestep_inputs)  # [20, 5]*[32, 1, 5, 128],  (32, 1, 20, 128)
        pre_msg = torch.cat([senders, receivers], dim=-1)         # (32, 1, 20, 256)
        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), 256)) # (32, 1, 20, 256)
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()
            
        start_idx = 0
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg)) # (32, 1, 20, 256)
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))     # (32, 1, 20, 256)
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            if i == 0:
                all_msgs = msg
            else:
                all_msgs = torch.cat((all_msgs, msg),3) # (32, 1, 20, 512)
        
        # Aggregate all msgs to receiver
        nod_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1) # (32, 1, 5, 512)
        
        agg_msgs   = F.relu(self.bn6(self.conv6(nod_msgs)))             #  (32, 64, 5, 512)
        merge_msgs = F.relu(self.bn7(self.conv7(agg_msgs)))             #  (32, 64, 1, 512) 
        query = F.relu(self.bnQ(self.Q(merge_msgs)))                    #  (32, 1,  1, 512)     
        key   = F.relu(self.bnK(self.K(agg_msgs)))                      #  (32, 1,  5, 512)
        value = F.relu(self.bnV(self.V(agg_msgs)))                      #  (32, 1,  5, 512)
        query = query.squeeze()                                         #  (32, 512)   
        key   = key.squeeze()                                           #  (32, 5, 512)
        value = value.squeeze()                                         #  (32, 5, 512)                       
        score = torch.matmul(key, query.unsqueeze(2))                   #  (32, 5, 1)
        score = F.softmax(score / math.sqrt(query.size(-1)), dim = 1)   #  (32, 5, 1)
        score = score.squeeze()                                         #  (32, 5)        
                
        weighted_features = torch.zeros(agg_msgs.shape[0], nod_msgs.shape[3]).cuda() #  (32, 512)
        for i in range(5):
            weighted_features.add_(torch.mul(value[:, i,:], score[:, i].reshape(len(score[:, i]), 1)))
        # weighted_features.shape (32, 512)
        ''' add '''
        weighted_features += query                                        #  (32, 512)   
        
        preds = weighted_features # (32, 512)
        preds = F.dropout(preds, p=self.dropout_prob, training=self.training)   
        preds = self.fc_out(preds)                # (32, 5)        
        
        return preds  # (32, 5)

    def forward(self, inputs, rel_rec, rel_send): # (32, 5, 300, 9)
    
        ''' feature extractor '''
        x = inputs.reshape(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3]) # (160, 300, 9)
        x = x.unsqueeze(1)                       #  (160, 1,  300, 9)     
        x = F.relu(self.bn1(self.conv1(x)))      #  (160, 64, 296, 1)
        x = x.squeeze()                          #  (160, 64, 296)
        x = self.pool(x)                         #  (160, 64, 148)
        x = F.relu(self.bn2(self.conv2(x)))      #  (160, 64, 144)
        x = self.pool(x)                         #  (160, 64, 72)
        x = F.relu(self.bn3(self.conv3(x)))      #  (160, 64, 68)
        x = self.pool(x)                         #  (160, 64, 34)
        
        x = x.permute(2, 0, 1)                   #  (34, 160, 64)
        x = F.dropout(x, p=self.dropout_prob, training=self.training) 
        x, (h_n,c_n) = self.lstm1(x)             #  (34, 160, 128)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x, (h_n,c_n) = self.lstm2(x)             #  (34, 160, 128)
        x = x[-1]                                #  (160, 128)
    
        x = x.reshape(inputs.shape[0], inputs.shape[1], -1) # (32, 5, 128)
        s_input = x
        
        ''' encoder '''
        x = self.node2edge(x, rel_rec, rel_send)            # (32, 20, 256)
        x = self.mlp1(x)                                    # (32, 20, 256)
        x_skip = x                                          # (32, 20, 256)
        x = self.edge2node(x, rel_rec, rel_send)            # (32, 5, 256)
        x = self.mlp2(x)                                    # (32, 5, 256)
        x = self.node2edge(x, rel_rec, rel_send)            # (32, 20, 512)
        x = torch.cat((x, x_skip), dim=2)                   # (32, 20, 768)
        x = self.mlp3(x)                                    # (32, 20, 64)
        logits = self.out(x)                                # (32, 20, 2)
        rel_type = my_softmax(logits, -1)                   # (32, 20, 2)
        
        ''' decoder '''        
        s_input = s_input.unsqueeze(1)   # (32, 1, 5, 128)
        rel_type = rel_type.unsqueeze(1) # (32, 1, 20, 2)
        preds= self.single_step_forward(s_input, rel_rec, rel_send, rel_type)   # (32, 5)
        
        return preds  # (32, 5)        