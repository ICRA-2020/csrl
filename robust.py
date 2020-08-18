#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from csrl.mdp import GridMDP
from csrl.oa import OmegaAutomaton
from csrl import ControlSynthesis
import numpy as np

from multiprocessing import cpu_count

T = 2**10
K = 2**20
print(T,K,min(32,cpu_count()))

ltl = 'F G a'
# Translate the LTL formula to an LDBA
oa = OmegaAutomaton(ltl,oa_type='dra')
print('Number of Omega-automaton states (including the trap state):',oa.shape[1])
# display(oa)


# MDP Description
shape = (5,4)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
    ['E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'T'],
    ['B',  'E',  'E',  'E'],
    ['E',  'E',  'B',  'E'],
    ['B',  'T',  'E',  'E']
])

label = np.array([
    [(),       ('a',),     (),    ()],
    [(),       (),     (),    ()],
    [(),       (),     (),    ()],
    [('a',),   (),     (),    ()],
    [(),       (),     (),    ()]
],dtype=np.object)

reward = np.zeros(shape)

grid_mdp = GridMDP(shape=shape,structure=structure,reward=reward,label=label,figsize=6,robust=True)  # Use figsize=4 for smaller figures
# grid_mdp.plot()
# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa)


# In[2]:


# oa.__dict__


# In[3]:


# Q=csrl.minimax_q(T=2**5,K=2**10)
# value = np.max(np.min(Q,axis=-1),axis=-1)
# policy, policy_ = csrl.get_greedy_policies(value)
# csrl.plot(value=value,policy=policy,policy_=policy_)


# In[4]:


# Specification
ltl = 'F G s & G F a & G F b'
oa = OmegaAutomaton(ltl,oa_type='dra')
print('Number of Omega-automaton states (including the trap state):',oa.shape[1])
# display(oa)

# MDP Description
shape = (6,5)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
    ['E',  'E',  'B',  'E',  'E'],
    ['E',  'E',  'B',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E']
])

label = np.array([
    [('a','s'), ('b','s'), (),     (),        ()],
    [(),        (),        (),     (),        ()],
    [('s',),    ('s',),    ('s',), ('s',),    ('s',)],
    [('s',),    ('s',),    ('s',), ('s',),    ('s',)],
    [('s',),    ('s',),    (),     ('s',),    ('s',)],
    [('s',),    ('a','s'), ('s',), ('b','s'), ('s',)]
],dtype=np.object)

reward = np.zeros(shape)

grid_mdp = GridMDP(shape=shape,structure=structure,reward=reward,label=label,figsize=6,robust=True)  # Use figsize=4 for smaller figures
# grid_mdp.plot()
# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa)


# In[5]:


oa.__dict__


# In[6]:


Q=csrl.minimax_q(T=T,K=K)
# value = np.max(np.min(Q,axis=-1),axis=-1)
# csrl.plot(value)


# In[7]:


# policy = np.argmax(np.min(Q,axis=-1),axis=-1)
# policy_ = np.take_along_axis(np.argmin(Q,axis=-1),np.expand_dims(policy,axis=-1),axis=-1).reshape(policy.shape)
# csrl.plot(policy=policy,policy_=policy_)


# In[8]:


np.save('Q-'+str((T-1).bit_length())+'-'+str((K-1).bit_length())+'-'+str(min(32,cpu_count())),Q)

