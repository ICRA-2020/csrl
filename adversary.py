#!/usr/bin/env python
# coding: utf-8

# In[1]:

from csrl.mdp import GridMDP
from csrl.oa import OmegaAutomaton
from csrl import ControlSynthesis
import numpy as np

from multiprocessing import cpu_count

T = 2**10
K = 2**15
print(T,K,min(32,cpu_count()))

# Specification
ltl = 'F G a & G !b'
oa = OmegaAutomaton(ltl,oa_type='dra')
print('Number of Omega-automaton states (including the trap state):',oa.shape[1])

# MDP Description
shape = (5,5)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
    ['E',  'E',  'T',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E'],
    ['T',  'B',  'T',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'T',  'E',  'E']
])

label = np.array([
    [(),        (),        (),        ('a',),        ('a',)],
    [(),        (),        (),        ('a',),        ('a',)],
    [('a',),    (),        (),        ('a',),        ('a',)],
    [(),        (),        (),        ('a',),        ('a',)],
    [(),        (),        (),        ('a',),        ('a',)],
],dtype=np.object)

reward = np.zeros(shape)

grid_mdp = GridMDP(shape=shape,structure=structure,reward=reward,label=label,figsize=6,second_agent=('b',))  # Use figsize=4 for smaller figures

# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa)


# In[2]:


print(oa.__dict__)


# In[3]:


Q,Q_=csrl.minimax_q(T=T,K=K,start=(0,4),start_=(4,0))


# In[4]:


# policy = np.argmax(Q,axis=-1)
# policy_ = np.argmin(Q_,axis=-1)
# value = np.max(Q,axis=-1)


# In[5]:


# episode=csrl.simulate(policy,policy_,start=(0,4),start_=(4,0),T=1000,plot=False)


# In[6]:


np.save('Q,Q_-'+str((T-1).bit_length())+'-'+str((K-1).bit_length())+'-'+str(min(32,cpu_count())),(Q,Q_))

