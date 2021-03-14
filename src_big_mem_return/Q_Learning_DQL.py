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

from snake import SnakeGame
from IPython.display import clear_output
import os
import matplotlib.animation as animation
import matplotlib.image as image
from matplotlib import rc


# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = "1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


# if gpu is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


#torch.cuda.device(1) #set gpu 0 or 1 or ...
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.current_device()


# In[5]:


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
    
    


# In[6]:


#not very flexible. Whatever for now.

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[up0exp,down0exp,left0..,right0]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.tanh(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# In[7]:


# Hyper parametr for NN
BATCH_SIZE = 128
GAMMA = 0.999
EPS = 0.05
TARGET_UPDATE = 10


# In[8]:


game_size = 8
n_actions = 4 # up down left right


# In[9]:


policy_net = DQN(game_size , game_size , n_actions).to(device)
target_net = DQN(game_size , game_size , n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


# In[10]:


#optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr=0.1)
memory = ReplayMemory(10000)


# In[11]:


def select_action(state):
    #print(policy_net(state),policy_net(state).max(1)[1].view(1, 1))
    #return policy_net(state).max(1)[1].view(1, 1)
    if random.random() > EPS:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long) 
    
episode_durations = []


# In[ ]:





# In[12]:


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# In[13]:


def get_board(game):
    board = np.asarray(game.getBoard(),dtype="float32")
    Max = max(game.get_parameter())
    board/= (Max/2)  #scale to 0-2
    board-= 1.0      #scale to -1-1
    board = board.reshape(1,game_size,game_size)
    board = torch.from_numpy(board)
    return board.unsqueeze(0).to(device)


# In[14]:


def AI_play(mode="easy",wait=0.1,size=game_size,cutoff=float("inf")):
    #print("I want to play a game")
    game = SnakeGame(size, size,mode)
    game.display()
    gameOver=False
    reward=0
    score=0
    print("Score:",score)
    ims = []
    step=0
    while True:
        step+=1
        if(step>cutoff):break
        clear_output(wait=True)
        action = select_action(get_board(game))
        _, _, gameOver, score = game.makeMove(action.item())
        
        
        if score==size**2:
            print("Game Over,Perfect Score:", score) #score  = lenghth of the snake
            break
        
        if gameOver:
            print("Game Over, Score:", score)
            break
        else:
            game.display()
            print("Score:", score)
        time.sleep(wait)
        ims+=[np.copy(game.getBoard())]
        #print(ims)
    return ims


# In[ ]:


num_episodes = 10**6
final_score_record=[]
for i_episode in range(num_episodes):
    # Initialize the environment and state
    mode = "hard"
    if(i_episode<10**5):mode = "easy"
    elif(i_episode<2*10**5):mode = "mid"
        
    game = SnakeGame(game_size, game_size,mode)
    state = get_board(game)
    last_reward=0
    score_record=[]
        
    pre_action=-float('inf')
    repeat=0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, _, gameOver, score =  game.makeMove(action.item()) #score = len of snake  
        score*=100
        score+=min(100,(len(score_record)-repeat*2)*4)
        if(pre_action+action.item()==2 or pre_action+action.item()==4):
            repeat+=1 #repeat up-down or left-right for beginning
        #print(pre_action,action.item(),repeat)
        pre_action=action.item()
        if(gameOver):score-=1000
        elif(len(score_record)>100 and score_record[-1]==score_record[-100]):
            if(game.get_len()==game_size**2):
                score+=1000 #perfect
            else: score-=1000 #stuck
            gameOver=True
            
        score_record+=[score]
        score = torch.tensor([score], device=device)
        
        
        # Observe new state
        if not gameOver:
            next_state =get_board(game)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, score)
        
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if gameOver:
            episode_durations.append(t + 1)
            break
    final_score_record+=[score_record[-1]]
    
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if i_episode % 1000 == 0:   
        AI_play(mode,wait=0.5)
    if i_episode % 100 == 0:
        print(i_episode,"/",num_episodes," final score", score.cpu().numpy()[0], " snake length", game.get_len(),repeat)
        


# In[ ]:


plt.plot(final_score_record,"o")


# In[ ]:


#for param in target_net.parameters():
#      print(param.data)


# In[ ]:


ims = AI_play(mode="hard",wait=0,cutoff=1000)


# In[ ]:


fig = plt.figure()
ims2= [[plt.imshow(im, animated=True)] for im in ims]
im_ani = animation.ArtistAnimation(fig, ims2, interval=200,blit=True, repeat_delay=False,repeat=False)
rc('animation', html='html5')
im_ani 


# In[ ]:


im_ani.save("snake_play.gif", writer = 'imagemagick',fps=1)


# In[ ]:




