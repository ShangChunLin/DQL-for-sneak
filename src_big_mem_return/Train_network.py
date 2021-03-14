#!/usr/bin/env python
# coding: utf-8

# # DQL for snake game
# using board (matrix) as input, CNN, so no reward nor state, only scores and gameover

# In[1]:


import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import time 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T


# In[2]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    


# In[3]:


#not very flexible. Whatever for now.

class DQN(nn.Module):

    def __init__(self, time_window, h, w, outputs):
        
        super(DQN, self).__init__()
        nfc =128 #num of final chanel
        
        self.conv1 = nn.Conv3d(1, 128,kernel_size=(1,5,5))#super big boundary
        self.conv2 = nn.Conv3d(128,128,kernel_size=(1,5,5))
        #self.max1 = nn.MaxPool3d((1,2,2))
        self.conv3 = nn.Conv3d(128,128,kernel_size=(1,3,3))
        self.conv4 = nn.Conv3d(128,nfc,kernel_size=(1,3,3))
        
        convw = 6
        convh = 6
        convtime_window = time_window
        linear_input_size = convw * convh * convtime_window* nfc
        self.head = nn.Linear(linear_input_size, outputs)
        
    def forward(self, x):
        
        #print(x.shape)
        #x = nn.functional.pad(x,(6,6,6,6,0,0), mode='constant', value=0)
        #print(x.shape)
        #x = nn.functional.pad(x,(1,1,1,1,0,0), mode='constant', value=0)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #x = self.max1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print(x.shape)
        #x = F.relu(self.bn4(self.conv4(x)))
        #x = F.relu(self.bn5(self.conv5(x)))
        return self.head(x.view(x.size(0), -1))


# In[4]:


def get_memory(memory_board,game,device):
    #print(memory_board.shape)
    time_window,_,game_size = memory_board.shape
    memory_board *= 0.5
    board = np.asarray(game.getBoard(),dtype="float32")
    Range = max(game.get_parameter())-min(game.get_parameter())
    board/= (Range/2)  #scale to 0-2
    board-= 1.0      #scale to -1-1
    memory_board = memory_board[1:,:,:]
    memory_board = np.append(memory_board,[board],axis=0)
    tmp = memory_board.reshape(1,time_window,game_size,game_size)
    tmp = torch.from_numpy(tmp).float()
    return tmp.unsqueeze(0).to(device),memory_board


# In[ ]:




