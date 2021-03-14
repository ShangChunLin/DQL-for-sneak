#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Basic snake game for use by reinforcement learning.
Run this script to test it out in the console
'''
import numpy as np
import random
import matplotlib.pyplot as plt

try:
    from IPython.display import clear_output
except:
    pass


# In[2]:


class BodyNode():
    def __init__(self, parent, x, y):
        self.parent = parent
        self.x = x
        self.y = y

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def setParent(self, parent):
        self.parent = parent

    def getPosition(self):
        return (self.x, self.y)
    
    def getIndex(self):
        return (self.y, self.x)


# In[3]:


class Snake():
    def __init__(self, x, y):
        self.head = BodyNode(None, x, y)
        self.tail = self.head

    def moveBodyForwards(self):
        currentNode = self.tail
        while currentNode.parent != None:
            parentPosition = currentNode.parent.getPosition()
            currentNode.setX(parentPosition[0])
            currentNode.setY(parentPosition[1])
            currentNode = currentNode.parent

    def move(self, direction):
        (oldTailX, oldTailY) = self.tail.getPosition()
        self.moveBodyForwards()
        headPosition = self.head.getPosition()
        if direction == 0:
            self.head.setY(headPosition[1] - 1)
        elif direction == 1:
            self.head.setX(headPosition[0] + 1)
        elif direction == 2:
            self.head.setY(headPosition[1] + 1)
        elif direction == 3:
            self.head.setX(headPosition[0] - 1)
        return (oldTailX, oldTailY, self.head.getPosition())

    def newHead(self, newX, newY):
        newHead = BodyNode(None, newX, newY)
        self.head.setParent(newHead)
        self.head = newHead
        
    def getHead(self):
        return self.head
    
    def getTail(self):
        return self.tail


# In[4]:


class SnakeGame():
    def __init__(self, width, height,mode="hard"):
        # arbitrary numbers to signify head, body, and food)
        # 0 for empty space
        self.headVal = 2
        self.bodyVal = 1
        self.boundaryVal = -1
        self.foodVal = 3
        
        self.width = width
        self.height = height
        self.mode = mode
        #print(mode,self.mode)
        self.board = np.zeros([height, width], dtype=int)
        
        self.board[0,:]=self.boundaryVal
        self.board[-1,:]=self.boundaryVal
        self.board[:,0]=self.boundaryVal
        self.board[:,-1]=self.boundaryVal
        #print(self.board)
        self.length = 1
        #self.pre_move=float('inf')
        startX = width//2
        startY = height//2

        self.board[startX, startY] = self.headVal
        self.snake = Snake(startX, startY)
        self.spawnFood()
        self.calcState()
#        print(self.board)
    def get_parameter(self):
        return (self.headVal,self.bodyVal,self.foodVal,self.boundaryVal)
    
    def get_len(self):
        return (self.length)
    
    def spawnFood(self):
        # spawn food at location not occupied by snake
        emptyCells = []
        #total_food=1
        if(self.mode=="hard"):
            total_food=1
        elif(self.mode=="mid"):
            total_food=(self.height-2)*(self.width-2)//4
        elif(self.mode=="easy"):
            total_food=(self.height-2)*(self.width-2)//2
        else:
            print("must choose a mode")
        for index, value in np.ndenumerate(self.board):
            if value != self.bodyVal and value != self.headVal and value != self.boundaryVal and value != self.foodVal:
                emptyCells.append(index)
            if value == self.foodVal:total_food-=1
                #emptyCells.append(index)
        if(emptyCells):
            for _ in range(total_food):
                self.foodIndex = random.choice(emptyCells)
                self.board[self.foodIndex] = self.foodVal
            
    def checkValid(self, direction):
        # check if move is blocked by wall
        newX, newY = self.potentialPosition(direction)
        if newX == 0 or newX == self.width-1:
            return False
        if newY == 0 or newY == self.height-1:
            return False
        # check if move is blocked by snake body
        tailIndex = self.snake.getTail().getIndex()
        if self.board[newY, newX] == self.bodyVal:
            #print([newY, newX],tailIndex,(newY, newX)==tailIndex)
            if(self.length>=4 and (newY, newX)==tailIndex): #4=mininum len of circle
                return True
            return False
        return True
    
    #def checkPerfect(self):
    #    if(self.length==self.width*self.height):
    #        return True
    #    return False

    def potentialPosition(self, direction):
        (newX, newY) = self.snake.getHead().getPosition()
        if direction == 0:
            newY -= 1
        elif direction == 1:
            newX += 1
        elif direction == 2:
            newY += 1
        elif direction == 3:
            newX -= 1
        return (newX, newY)

    def calcState(self):
        # state is as follows.
        # Is direction blocked by wall or snake?
        # Is food in this direction?
        # (top blocked, right blocked, down blocked, left blocked,
        # top food, right food, down food, left food)
        self.state = np.zeros(8, dtype=int)
        for i in range(4):
            self.state[i] = not self.checkValid(i)
        self.state[4:] = self.calcFoodDirection()

    def calcStateNum(self):
        # calculate an integer number for state
        # there will be 2^8 potential states but not all states are reachable
        stateNum = 0
        for i in range(8):
            stateNum += 2**i*self.state[i]
        return stateNum

    def calcFoodDirection(self):
        # food can be 1 or 2 directions eg. right and up
        # 0 is up, 1 is right, 2 is down, 3 is left
        foodDirections = np.zeros(4, dtype=int)
        dist = np.array(self.foodIndex) - np.array(self.snake.getHead().getIndex())
        if dist[0] < 0:
            # down
            foodDirections[0] = 1
        elif dist[0] > 0:
            # up
            foodDirections[2] = 1
        if dist[1] > 0:
            # right
            foodDirections[1] = 1
        elif dist[1] < 0:
            # left
            foodDirections[3] = 1
        return foodDirections

    def plottableBoard(self):
        #returns board formatted for animations
        board = np.zeros([self.width, self.height])
        currentNode = self.snake.tail
        count = 0
        while True:
            count += 1
            board[currentNode.getIndex()] = 0.2 + 0.8*count/self.length
            currentNode = currentNode.parent
            if currentNode == None:
                break
        board[self.foodIndex] = -1
        return board
        
        
    def display(self):
        print(self.board)
    
    def getBoard(self):
        return self.board

    
    def makeMove(self, direction):
        #if(self.pre_move + direction == 2 or self.pre_move + direction == 4):
        #    direction=self.pre_move
        #else:
        #    self.pre_move=direction
        gameOver = False
        if self.checkValid(direction):
            # set reward if moving in the right direction
            if self.calcFoodDirection()[direction] == 1:
                reward = 1
            else:
                reward = 0
            (headX, headY) = self.snake.getHead().getPosition()
            # set old head to body val
            self.board[headY, headX] = self.bodyVal

            # check if we got the fruit
            potX, potY = self.potentialPosition(direction)
            if self.board[potY, potX] == self.foodVal:
                # extend the snake
                self.snake.newHead(potX, potY)
                self.board[potY, potX] = self.headVal
                self.spawnFood()
                self.length += 1
                # if you want to give a higher reward for getting the fruit, uncomment below
                reward = 2
            else:
                # move the snake
                (oldTailX, oldTailY, newHead) = self.snake.move(direction)
                newHeadX, newHeadY = newHead
                self.board[oldTailY, oldTailX] = 0
                self.board[newHeadY, newHeadX] = self.headVal
        else:
            reward = -2
            gameOver = True
            
        self.calcState()
        return (self.calcStateNum(), reward, gameOver, self.length)


# In[5]:


def play_a_game(size=8,mode="easy"):
    #print("I want to play a game")
    game = SnakeGame(size, size,mode=mode)
    game.display()
    gameOver=False
    reward=0
    score=0
    print("Score:",score)
    while True:
        clear_output(wait=True)
        direction = input("Input Direction (w,a,s,d or q to quit): ")
        print("I want to play a game")
        if direction == 'w':
            new_state, reward, gameOver, score = game.makeMove(0)
        elif direction == 'a':
            new_state, reward, gameOver, score = game.makeMove(3)
        elif direction == 's':
            new_state, reward, gameOver, score = game.makeMove(2)
        elif direction == 'd':
            new_state, reward, gameOver, score = game.makeMove(1)
        elif direction == 'q':
            print("break game")
            break

        
        if score==size**2:
            print("Game Over,Perfect Score:", score) #score  = lenghth of the snake
            break
        
        if gameOver:
            print("Game Over, Score:", score)
            break
        else:
            game.display()
            print("Reward:", reward, "Score:", score)
    


# In[7]:


#play_a_game(10,mode="hard")


# In[ ]:





# In[ ]:




