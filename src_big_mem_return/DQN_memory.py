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

from Train_network import ReplayMemory,DQN,Transition,get_memory


# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


# Hyper parametr for NN
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.0
EPS_DECAY = 1000
TARGET_UPDATE = 10


# In[4]:


game_size = 16 +2 #(+2 for bundary)
time_window = 2
n_actions = 4 # up down left right


# In[5]:


policy_net = DQN(time_window, game_size , game_size , n_actions).to(device)
target_net = DQN(time_window, game_size , game_size , n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


# In[6]:


#torch.save(target_net, "policy_net.pth")


# In[7]:


#optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)
memory = ReplayMemory(100000)


# In[8]:



steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *         math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# In[ ]:





# In[9]:


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
    #print(loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-.1, .1)
    optimizer.step()


# In[10]:


num_episodes = 10**5
final_score_record=[]
final_len_record=[] 
for i_episode in range(num_episodes):
    # Initialize the environment and state
    mode = "hard"
    #if(i_episode<0.01*num_episodes):mode = "easy"
    #elif(i_episode<0.02*num_episodes):mode = "mid"
    #if(i_episode<0.01*num_episodes):mode = "mid"
        
    game = SnakeGame(game_size, game_size,mode)
    memory_board = np.zeros((time_window,game_size,game_size))
    
    
    score_record=[]
    len_record=[] 
    state,memory_board = get_memory(memory_board,game,device)
    #repeat=0
    pre_action=-float('inf')
    #print(state.shape)
    total_score = 0
    positive_reward = 2
    default_reward = -0.1
    neg_reward = -1
    perfect_reward = 10
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, _, gameOver, score =  game.makeMove(action.item()) #score = len of snake  
        len_record+=[score]
        
        perfect=False
        if((len(score_record)>1000 and score_record[-1]<=score_record[-1000]) or (
            len(len_record)>1000 and len_record[-1]==1) or (len(len_record)>1000 and len_record[-1]<=len_record[-1000]) ):
            if(game.get_len()==game_size**2):
                perfect=True
            gameOver=True
        score_record+=[score]
        # Observe new state
        if not gameOver:
            next_state,memory_board = get_memory(memory_board,game,device)
        else:
            next_state = None

        # Store the transition in memory
        reward=default_reward
        if(len(len_record)>1):
            reward+=(len_record[-1]>len_record[-2])*positive_reward
            reward+=(perfect)*perfect_reward
        reward+=(gameOver)*neg_reward
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)
        
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy_net)
        optimize_model()
        
        if gameOver:
            break
    final_score_record+=[score_record[-1]]
    final_len_record+=[len_record[-1]]
    
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if i_episode % 100 == 0:
        torch.save(policy_net.state_dict(), "policy_net.pth")
        print(i_episode,"/",num_episodes, " snake length=", game.get_len(),
              "total steps=",len(len_record),"total score=",final_score_record[-1])
        try:
            print(" avg snake length=", np.mean(final_len_record[-100:]))
            print(" best snake length=", np.max(final_len_record[-100:]))
        except:
            print(" avg snake length=", np.mean(final_len_record))
            print(" best snake length=", np.max(final_len_record))
        #final_len_record=[]
        np.save("final_len",final_len_record)


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




