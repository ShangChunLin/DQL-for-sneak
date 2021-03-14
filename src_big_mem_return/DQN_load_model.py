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
from IPython.display import Image
import os
import matplotlib.animation as animation
import matplotlib.image as image
from matplotlib import rc

from Train_network import ReplayMemory,DQN,Transition,get_memory


# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


final_len=np.load("final_len.npy")
step = 100
tmp = [np.mean(final_len[i:i+step]) for i in range(len(final_len)-step)]
plt.plot(tmp,"-")
plt.xlabel("# evolutions",size=20)
plt.ylabel("mean snake lenghth",size=20)


# In[4]:


game_size = 16 +2 #(+2 for bundary)
time_window = 2
n_actions = 4 # up down left right


# In[5]:


target_net = DQN(time_window, game_size , game_size , n_actions).to(device)
target_net.load_state_dict(torch.load("policy_net.pth"))
target_net.eval()


# In[6]:


def select_action(state):
    with torch.no_grad():
        return target_net(state).max(1)[1].view(1, 1)


# In[7]:


def AI_play(mode="easy",wait=0.1,size=game_size,cutoff=float("inf"),repeat=100):
    #print("I want to play a game")
    best=0
    best_ims=[]
    for _ in range(repeat):
        if(_%10==0):print(_)
        game = SnakeGame(size, size,mode)
        #game.display()
        gameOver=False
        reward=0
        score=0
        #print("Score:",score)
        ims = []
        step=0
        score_record= []
        memory_board = np.zeros((time_window,game_size,game_size))
        while True:
            step+=1
            if(step>cutoff):break
            #clear_output(wait=True)
            state,memory_board=get_memory(memory_board,game,device)
            action = select_action(state)
            _, _, gameOver, score = game.makeMove(action.item())

            if score==(size-2)**2:
                #print("Perfect") #score  = lenghth of the snake
                break

            score_record+=[score]
            if(len(score_record)>100 and score_record[-1]<=score_record[-100]):
                #print("stuck")
                break

            if gameOver:
                break
            else:
                pass
                #game.display()
                #print("Score:", score)

            #time.sleep(wait)
            tmp = np.asarray(np.copy(game.getBoard()),dtype="float32")

            #tmp+=1.0
            #tmp/=5.0
            #tmp*=255.0

            ims+=[tmp]
            #print(ims)
        if(score>best):
            best=score
            best_ims=ims[:]
    print("Game Over, best Score:", best)
    print("frame:", len(best_ims))
    return best_ims


# In[8]:


ims = AI_play(mode="hard",wait=0,cutoff=1000)


# In[14]:




#plt.cla()
#plt.clf()
get_ipython().system('mkdir ani_data ')
get_ipython().system('rm ani_data/*.png')
for i,im in enumerate(ims):
    fig = plt.figure()
    plt.ion()
    plt.axis('off')
    plt.gca().set_aspect('equal')

    x,y = np.where(im==0)
    plt.scatter(x,y,c="w",marker="s",s=400, animated=True)
    
    x,y = np.where(im==2)
    plt.scatter(x,y,c="r",marker="o",s=200, animated=True)
    
    x,y = np.where(im==1)
    plt.scatter(x,y,c="g",marker="o",s=200, animated=True)
    
    x,y = np.where(im==-1)
    plt.scatter(x,y,c="k",marker="s",s=200, animated=True)
    
    x,y = np.where(im==4)
    plt.scatter(x,y,c="b",marker="*",s=200, animated=True)
    
    plt.savefig('ani_data/step'+str(i).zfill(5)+'.png', dpi = 128, bbox_inches="tight")
    
    #plt.clf()


# In[15]:


get_ipython().system('convert -delay 10 ani_data/*.png snake_animation.gif')
get_ipython().system('rm ani_data/*.png')
get_ipython().system('rm -r ani_data')


# In[16]:


with open('snake_animation.gif','rb') as file:
    display(Image(file.read()))
    


# In[ ]:




