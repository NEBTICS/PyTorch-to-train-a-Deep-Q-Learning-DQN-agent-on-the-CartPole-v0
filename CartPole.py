# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 18:33:18 2021

@author: smithbarbose

"""
'''All modules'''
import gym 
import math
import random 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from collections import namedtuple,deque
from itertools import count
from PIL import Image
'''Torch module'''
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env=gym.make('CartPole-v0').unwrapped

#setting up matplotlib
is_ipython='inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display 
plt.ion()

'''Gpu check'''
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(device)   



'''Replay memory'''
Transition=namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory=deque([],maxlen=capacity)
    def push(self,*args):
        #save a transition
        self.memory.append(Transition(*args))
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

'''Q network '''
class DQN(nn.Module):
    
    def __init__(self,h,w,outputs):
        super(DQN,self).__init__()
        self.conv1=nn.Conv2d(3, 16, kernel_size=5,stride=2)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16, 32, kernel_size=5,stride=2)
        self.bn2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32, 32, kernel_size=5,stride=2)
        self.bn3=nn.BatchNorm2d(32)
        #number of linear inputs depends upon the outputs of conv2d layer 
        #and therefor the input image size 
        def conv2d_size_output(size,kernel_size=5,stride=2):
            return (size - (kernel_size - 1)-1)// stride +1 
        convw=conv2d_size_output(conv2d_size_output(conv2d_size_output(w)))
        convh=conv2d_size_output(conv2d_size_output(conv2d_size_output(h)))        
        linear_input_size=convw*convh*32
        self.head=nn.Linear(linear_input_size,outputs)
    
    #feed forward network
    def forward(self,x):
        x=x.to(device)
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0),-1))
    
#input extraction

resize=T.Compose([T.ToPILImage(),T.Resize(40,interpolation=Image.CUBIC),T.ToTensor()])

def get_cart_location(screen_width):
    world_width=env.x_threshold *2 
    scale=screen_width/world_width
    return int(env.state[0]*scale + screen_width/2.0)#middle of cart

def get_screen():
    #return image 
    screen=env.render(mode='rgb_array').transpose((2,0,1))
    #if cart is in the lower half,strip off the top
    _, screen_height,screen_width=screen.shape
    screen=screen[:,int(screen_height*0.4):int(screen_height*0.8)]
    view_width=int(screen_width*0.6)
    cart_location=get_cart_location(screen_width)
    if cart_location<view_width//2:
        slice_range=slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range=slice(-view_width,None)
        
    else:
        slice_range=slice(cart_location-view_width // 2,
                          cart_location+view_width // 2)
    screen=screen[:,:,slice_range]
    screen=np.ascontiguousarray(screen,dtype=np.float32)/255
    screen=torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1,2,0).numpy(),interpolation='none')
plt.title("Example")
plt.show()


'''Training '''
BATCH_SIZE=128
GAMMA=0.999
EPS_START=0.9
EPS_END=0.05
EPS_DECAY=200
TARGET_UPDATE=10

#GET THE SCREEN SIZE 
init_screen=get_screen()
_,_,screen_height,screen_width=init_screen.shape      
#get number of actions form gym action space
n_actions=env.action_space.n

#defining the policy net

policy_net=DQN(screen_height, screen_width, n_actions).to(device)
target_net=DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer
optimizer = optim.RMSprop(policy_net.parameters())
memory=ReplayMemory(10000)

steps_done=0

def select_action(state):
    global steps_done
    sample=random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)  
    
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]],device=device,dtype=torch.long)
episode_duration=[]

def plot_duration() :
    plt.figure(2)
    plt.clf()
    durations_t=torch.tensor(episode_duration,dtype=torch.float)
    plt.title('Training....')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode average and plot them too
    if len(durations_t)>=100:
        means=durations_t.unflod(0,100,1).mean(1).view(-1)
        means=torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        

'''Training loop'''
def optimizer_model():
    if len(memory)<BATCH_SIZE:
        return
    transitions=memory.sample(BATCH_SIZE)
    #Transpose the batch 
    #thise converts the batch of array of Transitions to Transition of batch array
    batch=Transition(*zip(*transitions))
    #compute a mask of non-final states
    non_final_mask=torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)),device=device,
                                dtype=torch.bool)
    non_final_next_state=torch.cat([s for s in batch.next_state if s is not None ])
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    reward_batch=torch.cat(batch.reward)
    #Computing Q(s_t,a)
    # the model computesQ(s_t)
    state_action_values=policy_net(state_batch).gather(1,action_batch)
    #compute V(s_{t+1})
    next_state_values=torch.zeros(BATCH_SIZE,device=device)
    next_state_values[non_final_mask]=target_net(non_final_next_state).max(1)[0].detach()
    #compute the extended q values
    expected_state_action_values=(next_state_values + GAMMA)+reward_batch
    
    #computing the huber loss
    criterion=nn.SmoothL1Loss()
    loss=criterion(state_action_values,expected_state_action_values.unsqueeze(1))
    
    #optimizer.zero_grand()
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()

'''Loops'''
num_episodes=100
for i_episodes in range(num_episodes):
    #initilizing the enviroment
    env.reset()
    last_screen=get_screen()
    current_screen=get_screen()
    state=current_screen-last_screen
    for t in count():
        #select and perform an action
        action=select_action(state)
        _,reward,done,_=env.step(action.item())
        reward=torch.tensor([reward],device=device)
        
        #Oberver new state
        last_screen=current_screen
        current_screen=get_screen()
        if not done:
            next_state=current_screen - last_screen
        else:
            next_state=None
        #Store the transition in memory
        memory.push(state,action,next_state,reward)
        #move to next state
        state=next_state
        #perform one step of the optimization (on policy netwok)
        optimizer_model()
        if done:
            episode_duration.append(t+1)
            plot_duration()
            break
        
    #Update the target network,copyting all weights and biases in dqn
    if i_episodes % TARGET_UPDATE==0:
        target_net.load_state_dict(policy_net.state_dict())
print('************Completed**************')
env.render()
env.close()
plt.ioff()
plt.show()
    