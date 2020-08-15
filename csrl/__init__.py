"""Control Synthesis using Reinforcement Learning.
"""
import numpy as np
from itertools import product
from .oa import OmegaAutomaton
import os
from multiprocessing import shared_memory, Pool, cpu_count

import importlib

if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt

if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact

import cProfile, pstats, io
from pstats import SortKey

import random

class ControlSynthesis:
    """This class is the implementation of our main control synthesis algorithm.

    Attributes
    ----------
    shape : (n_pairs, n_qs, n_rows, n_cols, n_actions)
        The shape of the product MDP.

    reward : array, shape=(n_pairs,n_qs,n_rows,n_cols)
        The reward function of the star-MDP. self.reward[state] = 1-discountB if 'state' belongs to B, 0 otherwise.

    transition_probs : array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions)
        The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.

    Parameters
    ----------
    mdp : mdp.GridMDP
        The MDP that models the environment.

    oa : oa.OmegaAutomatan
        The OA obtained from the LTL specification.

    discount : float
        The discount factor.

    discountB : float
        The discount factor applied to B states.

    """

    def __init__(self, mdp, oa=None, discount=0.999999, discountB=0.9999, discountC=0.99):
        self.mdp = mdp
        self.oa = oa if oa else OmegaAutomaton(' | '.join([ap+' | !'+ap for ap in (mdp.AP+set(mdp.second_agent))]))
        self.discount = discount
        self.discountB = discountB  # We can also explicitly define a function of discount
        self.discountC = discountC  # Same
        self.shape = self.oa.shape + mdp.shape + (len(mdp.A)+self.oa.shape[1],)

        # Create the action matrix
        self.A = np.empty(self.shape[:-1],dtype=np.object)
        for i,q,r,c in self.states():
            self.A[i,q,r,c] = list(range(len(mdp.A))) + [len(mdp.A)+e_a for e_a in self.oa.eps[q]]

        # Create the reward matrix
        self.reward = np.zeros(self.shape[:-1])
        for i,q,r,c in self.states():
            acc_type = self.oa.acc[q][mdp.label[r,c]][i]
            self.reward[i,q,r,c] = 1-self.discountB if acc_type else (-1e-10 if acc_type is False else 0)

        # Create the transition matrix
        if mdp.robust:
            self.transition_probs = np.empty(self.shape+(len(self.mdp.A),),dtype=np.object)  # Enrich the action set with epsilon-actions
            for i,q,r,c in self.states():
                for action in self.A[i,q,r,c]:
                    for action_ in range(len(self.mdp.A)):
                        if action < len(self.mdp.A):  # MDP actions
                            q_ = self.oa.delta[q][mdp.label[r,c]]  # OA transition
                            mdp_states, probs = mdp.get_transition_prob((r,c),mdp.A[action],mdp.A[action_])  # MDP transition
                            self.transition_probs[i,q,r,c][action][action_] = [(i,q_,)+s for s in mdp_states], probs
                        else:  # epsilon-actions
                            self.transition_probs[i,q,r,c][action][action_] = ([(i,action-len(mdp.A),r,c)], [1.])

        elif mdp.second_agent:
            self.transition_probs = np.empty(mdp.shape+(len(self.mdp.A),),dtype=np.object)
            for r,c in self.states(short=True):
                for action in range(len(self.mdp.A)):
                    self.transition_probs[r,c][action] = mdp.get_transition_prob((r,c),mdp.A[action])
        else:
            self.transition_probs = np.empty(self.shape,dtype=np.object)
            for i,q,r,c in self.states():
                for action in self.A[i,q,r,c]:
                    if action < len(self.mdp.A):  # MDP actions
                        q_ = self.oa.delta[q][mdp.label[r,c]]  # OA transition
                        mdp_states, probs = mdp.get_transition_prob((r,c),mdp.A[action])  # MDP transition
                        self.transition_probs[i,q,r,c][action] = [(i,q_,)+s for s in mdp_states], probs
                    else:  # epsilon-actions
                        self.transition_probs[i,q,r,c][action] = ([(i,action-len(mdp.A),r,c)], [1.])

    def states(self,second=None,short=None):
        """State generator.

        Yields
        ------
        state: tuple
            State coordinates (i,q,r,c)).
        """
        n_pairs, n_qs, n_rows, n_cols, n_actions = self.shape
        if second:
            for i,q,r1,c1,r2,c2 in product(range(n_pairs),range(n_qs),range(n_rows),range(n_cols),range(n_rows),range(n_cols)):
                yield i,q,r1,c1,r2,c2
        elif short:
            for r,c in product(range(n_rows),range(n_cols)):
                yield r,c
        else:
            for i,q,r,c in product(range(n_pairs),range(n_qs),range(n_rows),range(n_cols)):
                yield i,q,r,c


    def random_state(self):
        """Generates a random state coordinate.

        Returns
        -------
        state: tuple
            A random state coordinate (i,q,r,c).
        """
        n_pairs, n_qs, n_rows, n_cols, n_actions = self.shape
        mdp_state = np.random.randint(n_rows),np.random.randint(n_cols)
        return (np.random.randint(n_pairs),np.random.randint(n_qs)) + mdp_state

    def q_learning(self,start=None,T=None,K=None):
        """Performs the Q-learning algorithm and returns the action values.

        Parameters
        ----------
        start : int
            The start state of the MDP.

        T : int
            The episode length.

        K : int
            The number of episodes.

        Returns
        -------
        Q: array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions) 
            The action values learned.
        """

        T = T if T else np.prod(self.shape[:-1])
        K = K if K else 100000

        Q = np.zeros(self.shape)

        for k in range(K):
            state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
            alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
            epsilon = np.max((1.0*(1 - 1.5*k/K),0.01))
            for t in range(T):

                reward = self.reward[state]
                gamma = self.discountB if reward else self.discount

                # Follow an epsilon-greedy policy
                if np.random.rand() < epsilon or np.max(Q[state])==0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])

                # Observe the next state
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states),p=probs)]

                # Q-update
                Q[state][action] += alpha * (reward + gamma*np.max(Q[next_state]) - Q[state][action])

                state = next_state

        return Q

    def greedy_policy(self, value):
        """Returns a greedy policy for the given value function.

        Parameters
        ----------
        value: array, size=(n_pairs,n_qs,n_rows,n_cols)
            The value function.

        Returns
        -------
        policy : array, size=(n_pairs,n_qs,n_rows,n_cols)
            The policy.

        """
        policy = np.zeros((value.shape),dtype=np.int)
        for state in self.states():
            action_values = np.empty(len(self.A[state]))
            for i,action in enumerate(self.A[state]):
                action_values[i] = np.sum([value[s]*p for s,p in zip(*self.transition_probs[state][action])])
            policy[state] = self.A[state][np.argmax(action_values)]
        return policy

    def value_iteration(self, T=None, threshold=None):
        """Performs the value iteration algorithm and returns the value function. It requires at least one parameter.

        Parameters
        ----------
        T : int
            The number of iterations.

        threshold: float
            The threshold value to be used in the stopping condition.

        Returns
        -------
        value: array, size=(n_pairs,n_qs,n_rows,n_cols)
            The value function.
        """
        value = np.zeros(self.shape[:-1])
        old_value = np.copy(value)
        t = 0  # The time step
        d = np.inf  # The difference between the last two steps
        while (T and t<T) or (threshold and d>threshold):
            value, old_value = old_value, value
            for state in self.states():
                # Bellman operator
                action_values = np.empty(len(self.A[state]))
                for i,action in enumerate(self.A[state]):
                    action_values[i] = np.sum([old_value[s]*p for s,p in zip(*self.transition_probs[state][action])])
                gamma = self.discountB if self.reward[state]>0 else self.discount
                value[state] = self.reward[state] + gamma*np.max(action_values)
            t += 1
            d = np.nanmax(np.abs(old_value-value))

        return value

    def simulate(self, policy, policy_=None, value=None, start=None, start_=None, T=None, plot=True, animation=None):
        """Simulates the environment and returns a trajectory obtained under the given policy.

        Parameters
        ----------
        policy : array, size=(n_pairs,n_qs,n_rows,n_cols)
            The policy.

        start : int
            The start state of the MDP.

        T : int
            The episode length.

        plot : bool
            Plots the simulation if it is True.

        Returns
        -------
        episode: list
            A sequence of states
        """
        T = T if T else np.prod(self.shape[:-1])
        if policy_ is None:
            state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
            episode = [state]
            for t in range(T):
                states, probs = self.transition_probs[state][policy[state]]
                state = states[np.random.choice(len(states),p=probs)]
                episode.append(state)

            if plot:
                def plot_agent(t):
                    self.mdp.plot(policy=policy[episode[t][:2]],agent=episode[t][2:])
                t=IntSlider(value=0,min=0,max=T-1)
                interact(plot_agent,t=t)

            if animation:
                pad=5
                if not os.path.exists(animation):
                    os.makedirs(animation)
                for t in range(T):
                    self.mdp.plot(value=value[episode[t][:2]],policy=policy[episode[t][:2]],agent=episode[t][2:],save=animation+os.sep+str(t).zfill(pad)+'.png',title='Time: '+str(t)+',  LDBA State (Mode): '+str(episode[t][1]))
                    plt.close()
                os.system('ffmpeg -r 3 -i '+animation+os.sep+'%0'+str(pad)+'d.png -vcodec libx264 -y '+animation+'.mp4')

            return episode

        else:

            k,q = (self.shape[0]-1,self.oa.q0)
            s1 = start if start else self.mdp.random_state()
            s2 = start_ if start_ else self.mdp.random_state()
            episode = [(k,q)+s1+s2]
            for t in range(T):

                states, probs = self.mdp.get_transition_prob(s1,self.mdp.A[policy[(k,q)+s1+s2]])
                s1 = states[np.random.choice(len(states),p=probs)]

                label = self.mdp.label[s1]
                if s1 == s2:
                    label += self.mdp.second_agent
                q = self.oa.delta[q][label]  # OA transition

                states, probs = self.mdp.get_transition_prob(s2,self.mdp.A[policy_[(k,q)+s1+s2]])
                s2 = states[np.random.choice(len(states),p=probs)]

                episode.append((k,q)+s1+s2)

            if plot:
                def plot_agent(t):
                    k,q,r1,c1,r2,c2 = episode[t]
                    self.mdp.plot(policy=policy[k,q,:,:,r2,c2],agent=(r1,c1),agent_=(r2,c2))
                t=IntSlider(value=0,min=0,max=T-1)
                interact(plot_agent,t=t)

            return episode

    def plot(self, value=None, policy=None, policy_=None, iq=None, **kwargs):
        """Plots the values of the states as a color matrix with two sliders.

        Parameters
        ----------
        value : array, shape=(n_pairs,n_qs,n_rows,n_cols)
            The value function.

        policy : array, shape=(n_pairs,n_qs,n_rows,n_cols)
            The policy to be visualized. It is optional.

        save : str
            The name of the file the image will be saved to. It is optional
        """

        if iq:
            val = value[iq] if value is not None else None
            pol = policy[iq] if policy is not None else None
            pol_ = policy_[iq] if policy_ is not None else None
            self.mdp.plot(val,pol,pol_,**kwargs)
        else:
            if self.mdp.second_agent:
                # A helper function for the sliders
                def plot_value(i,q,r1,c1,r2,c2):
                    val = value[i,q,:,:,r2,c2] if value is not None else None
                    pol = policy[i,q,:,:,r2,c2] if policy is not None else None
                    pol_ = policy_[i,q,r1,c1,:,:] if policy_ is not None else None
                    self.mdp.plot(val,pol,pol_,**kwargs)
                i = IntSlider(value=0,min=0,max=self.shape[0]-1)
                q = IntSlider(value=self.oa.q0,min=0,max=self.shape[1]-1)
                r1 = IntSlider(value=0,min=0,max=self.mdp.shape[0]-1)
                c1 = IntSlider(value=0,min=0,max=self.mdp.shape[1]-1)
                r2 = IntSlider(value=0,min=0,max=self.mdp.shape[0]-1)
                c2 = IntSlider(value=0,min=0,max=self.mdp.shape[1]-1)
                interact(plot_value,i=i,q=q,r1=r1,c1=c1,r2=r2,c2=c2)
            else:
                # A helper function for the sliders
                def plot_value(i,q):
                    val = value[i,q] if value is not None else None
                    pol = policy[i,q] if policy is not None else None
                    pol_ = policy_[i,q] if policy_ is not None else None
                    self.mdp.plot(val,pol,pol_,**kwargs)
                i = IntSlider(value=0,min=0,max=self.shape[0]-1)
                q = IntSlider(value=self.oa.q0,min=0,max=self.shape[1]-1)
                interact(plot_value,i=i,q=q)

    def get_greedy_policies(self, value):

        policy = np.zeros((value.shape),dtype=np.int)
        policy_ = np.zeros((value.shape),dtype=np.int)
        action_values_ = np.empty(len(self.mdp.A))

        if not self.mdp.second_agent:
            for state in self.states():
                # Bellman operator
                action_values = np.zeros(len(self.A[state]))
                action_mins = np.zeros(len(self.A[state]),dtype=np.int)
                for i,action in enumerate(self.A[state]):
                    for j,action_ in enumerate(range(len(self.mdp.A))):
                        action_values_[j] = np.sum([value[s]*p for s,p in zip(*self.transition_probs[state][action][action_])])
                    action_values[i] = np.min(action_values_)
                    action_mins[i] = np.argmin(action_values_)
                action_max = np.argmax(action_values)
                action_min = action_mins[action_max]
                policy[state] = self.A[state][action_max]
                policy_[state] = self.A[state][action_min]

        else:
            for k,q,r1,c1,r2,c2 in self.states(second=self.mdp.second_agent):
                s1 = r1,c1
                s2 = r2,c2

                label = self.mdp.label[s1]
                if s1 == s2:
                    label += self.mdp.second_agent
                q_ = self.oa.delta[q][label]  # OA transition

                # Bellman operator
                action_values = np.zeros(len(self.A[k,q,r1,c1]))
                for i,action in enumerate(self.A[k,q,r1,c1]):
                    # for s, p in zip(*self.mdp.get_transition_prob(s1,self.mdp.A[action])):
                    for s, p in zip(*self.transition_probs[s1][action]):
                        for j,action_ in enumerate(range(len(self.mdp.A))):
                            # action_values_[j] = np.sum([value[k,q_][s][s_]*p_ for s_,p_ in zip(*self.mdp.get_transition_prob(s2,self.mdp.A[action_]))])
                            action_values_[j] = np.sum([value[k,q_][s][s_]*p_ for s_,p_ in zip(*self.transition_probs[s2][action_])])
                        action_values[i] += np.min(action_values_)*p
                        policy_[k,q_][s][s2] = np.argmin(action_values_)

                policy[k,q][s1][s2] = self.A[k,q,r1,c1][np.argmax(action_values)]

        return policy, policy_


    def shapley(self, T=None, threshold=None):
        """Performs the Shapley's algorithm and returns the value function. It requires at least one parameter.

        Parameters
        ----------
        T : int
            The number of iterations.

        threshold: float
            The threshold value to be used in the stopping condition.

        Returns
        -------
        value: array, size=(n_pairs,n_qs,n_rows,n_cols)
            The value function.
        """
        # d = np.inf  # The difference between the last two steps
        shape = self.shape[:-1] if not self.mdp.second_agent else self.oa.shape+self.mdp.shape+self.mdp.shape
        value = np.zeros(shape)
        shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
        states = list(self.states(second=self.mdp.second_agent))
        m = cpu_count()
        n = len(states) // m
        arg_list = [[self,states[i*n:(i+1)*n],T,shape,shm.name] for i in range(m)]
        arg_list[-1][1] = states[(m-1)*n:]
        with Pool(m) as p:
            stats = p.map(shapley_iteration,arg_list)
        value = np.copy(np.ndarray(shape, dtype=np.float64, buffer=shm.buf))
        shm.close()
        shm.unlink()
        return value, stats
    
    def minimax_q(self,start=None,start_=None,T=None,K=None):
        n_actions = len(self.mdp.A)
        actions = list(range(n_actions))

        if not self.mdp.second_agent:
            shape = self.oa.shape + self.mdp.shape + (n_actions,n_actions)
            Q = np.zeros(shape)
            shm = shared_memory.SharedMemory(create=True, size=Q.nbytes)
            m = cpu_count()
            arg_list = [[self,T,K,start,shape,shm.name] for i in range(m)]
            with Pool(m) as p:
                p.map(minimax_q_robust,arg_list)
            Q = np.copy(np.ndarray(shape, dtype=np.float64, buffer=shm.buf))
            shm.close()
            shm.unlink()
            return Q
        else:
            shape = self.oa.shape + self.mdp.shape + self.mdp.shape + (n_actions,)
            Q = np.zeros(shape)
            Q_ = np.zeros(shape)
            shm = shared_memory.SharedMemory(create=True, size=Q.nbytes)
            shm_ = shared_memory.SharedMemory(create=True, size=Q_.nbytes)
            m = cpu_count()
            arg_list = [[self,T,K,start,start_,shape,shm.name,shm_.name] for i in range(m)]
            with Pool(m) as p:
                p.map(minimax_q_two_player,arg_list)
            Q = np.copy(np.ndarray(shape, dtype=np.float64, buffer=shm.buf))
            Q_ = np.copy(np.ndarray(shape, dtype=np.float64, buffer=shm_.buf))
            shm.close()
            shm.unlink()
            shm_.close()
            shm_.unlink()
            return Q, Q_


def minimax_q_two_player(arg_list):
    self,T,K,start,start_,shape,shm_name,shm_name_ = arg_list
    shm = shared_memory.SharedMemory(name=shm_name)
    Q = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    
    shm_ = shared_memory.SharedMemory(name=shm_name_)
    Q_ = np.ndarray(shape, dtype=np.float64, buffer=shm_.buf)
    
    n_actions = len(self.mdp.A)
    actions = list(range(n_actions))
    
    for k in range(K):
        state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
        alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
        epsilon = np.max((1.0*(1 - 1.05*k/K),0.01))

        k,q = (self.shape[0]-1,self.oa.q0)
        s1 = start if start else self.mdp.random_state()
        s2 = start_ if start_ else self.mdp.random_state()

        label = self.mdp.label[s1]
        if s1 == s2:
            label += self.mdp.second_agent
        q = self.oa.delta[q][label]  # OA transition

        max_action = random.randrange(n_actions)
        max_q = 0
        for t in range(T):

            if random.random() < epsilon:
                max_action = random.randrange(n_actions)
                max_q = Q[k,q][s1][s2][max_action]

            states, probs = self.transition_probs[s1][max_action]
            next_s1 = random.choices(states,weights=probs)[0]

            min_q, min_action = 1, 0
            for action_ in range(n_actions):
                if Q_[k,q][next_s1][s2][action_] < min_q:
                    min_action = action_
                    min_q = Q_[k,q][next_s1][s2][action_] 

            acc_type = self.oa.acc[q][label][k]
            reward = 0
            gamma = self.discount
            if acc_type is True:
                reward = 1-self.discountB
                gamma = self.discountB
            elif acc_type is False:
                gamma = self.discountC

            Q[k,q][s1][s2][max_action] = min(max_q + alpha * (reward + gamma*min_q - max_q), 1)

            if random.random() < epsilon:
                min_action = random.randrange(n_actions)
                min_q = Q_[k,q][next_s1][s2][min_action]

            states, probs = self.transition_probs[s2][min_action]
            next_s2 = random.choices(states,weights=probs)[0]

            label = self.mdp.label[next_s1]
            if next_s1 == next_s2:
                label += self.mdp.second_agent
            next_q = self.oa.delta[q][label]  # OA transition

            max_q, max_action = 0, 0
            for action in range(n_actions):
                if Q[k,next_q][next_s1][next_s2][action] > max_q:
                    max_action = action
                    max_q = Q[k,next_q][next_s1][next_s2][action]

            Q_[k,q][next_s1][s2][min_action] = min(min_q + alpha * (max_q - min_q), 1)

            q,s1,s2 = next_q, next_s1, next_s2

    
def minimax_q_robust(arg_list):
    self,T,K,start,shape,shm_name = arg_list
    shm = shared_memory.SharedMemory(name=shm_name)
    Q = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    
    n_actions = len(self.mdp.A)
    actions = list(range(n_actions))
    for k in range(K):
        state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
        alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
        epsilon = np.max((1.0*(1 - 1.05*k/K),0.01))
        for t in range(T):

                # Follow an epsilon-greedy policy
                if random.random() < epsilon:
                    max_action = random.randrange(n_actions)
                    min_action = random.randrange(n_actions)
                    max_q = Q[state][max_action][min_action]
                else:
                    max_action, max_q = 0, 0
                    min_action = 0
                    for i in range(n_actions):
                        action_, min_q = 0, 1
                        for j in range(n_actions):
                            if Q[state][i][j] < min_q:
                                action_ = j
                                min_q = Q[state][i][j]
                        if min_q > max_q:
                            min_action = action_
                            max_action = i
                            max_q = min_q



                # Observe the next state
                states, probs = self.transition_probs[state][max_action][min_action]
                next_state = random.choices(states,weights=probs)[0]

                next_max_q = 0
                for i in range(n_actions):
                    next_min_q = 1
                    for j in range(n_actions):
                        if Q[next_state][i][j] < next_min_q:
                            next_min_q = Q[next_state][i][j]
                    if next_min_q > next_max_q:
                        next_max_q = next_min_q

                reward = self.reward[state]
                gamma = self.discountB if self.reward[state]>0 else (self.discountC if self.reward[state]<0 else self.discount)

                # Q-update
                Q[state][max_action][min_action] += alpha * (reward + gamma*next_max_q - max_q)

                state = next_state

def shapley_iteration(args):
    pr = cProfile.Profile()
    pr.enable()

    self,states,T,shape,shm_name = args
    shm = shared_memory.SharedMemory(name=shm_name)
    value = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    t = 0  # The time step
    action_values_ = np.empty(len(self.mdp.A))
#     old_value = np.copy(value)
    while t < T:
        for state in states:
            if not self.mdp.second_agent:
                # Bellman operator
                action_values = np.empty(len(self.A[state]))
                action_max = 0
                for i,action in enumerate(self.A[state]):
                    action_min = 1
                    for j,action_ in enumerate(range(len(self.mdp.A))):
                        val = 0
                        for s,p in zip(*self.transition_probs[state][action][action_]):
                            val += value[s]*p
#                         action_values_[j] = np.sum([value[s]*p for s,p in zip(*self.transition_probs[state][action][action_])])
                        action_min = min(action_min,val)
                    action_max= max(action_max,action_min)
#                     action_values[i] = np.min(action_values_)
                gamma = self.discountB if self.reward[state]>0 else (self.discountC if self.reward[state]<0 else self.discount)
                value[state] = self.reward[state] + gamma*action_max  # np.max(action_values)

            else:
                k,q,r1,c1,r2,c2 = state
                s1 = r1,c1
                s2 = r2,c2

                label = self.mdp.label[s1]
                if s1 == s2:
                    label += self.mdp.second_agent
                q_ = self.oa.delta[q][label]  # OA transition

                # Bellman operator
                action_values = np.zeros(len(self.A[k,q,r1,c1]))
                for i,action in enumerate(self.A[k,q,r1,c1]):
                    # for s, p in zip(*self.mdp.get_transition_prob(s1,self.mdp.A[action])):
                    for s, p in zip(*self.transition_probs[s1][action]):
                        action_min = 1
                        for j,action_ in enumerate(range(len(self.mdp.A))):
                            val=0
                            for s_,p_ in zip(*self.transition_probs[s2][action_]):
                                val += value[k,q_][s][s_]*p_
                            action_min = min(action_min,val)
                            # action_values_[j] = np.sum([value[k,q_][s][s_]*p_ for s_,p_ in zip(*self.mdp.get_transition_prob(s2,self.mdp.A[action_]))])
                        action_values[i] += action_min*p

                acc_type = self.oa.acc[q][label][k]
                reward = 0
                gamma = self.discount
                if acc_type is True:
                    reward = 1-self.discountB
                    gamma = self.discountB
                elif acc_type is False:
                    gamma = self.discountC

                value[state] = reward + gamma*np.max(action_values)
        t+=1
#     shm.close()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    return s.getvalue()
