"""Grid World MDP Implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets.widgets import IntSlider
from ipywidgets import interact
from itertools import product
from matplotlib.font_manager import FontProperties

# Down, Up, Right, Left
Actions = ['D','U','R','L']  

class GridMDP():
    """This class implements a Markov Decision Process where an agent can move up, down, right or left in a 2D grid world.
    
    Attributes
    ----------
    shape : (n_rows, n_cols)
        The shape of the grid.
        
    delta : list of dicts
        The transition function of the DRA. delta[q][label_set] is the number of the state that the DRA makes a transition to when it consumes the label_set in the state q.
        
    acc : array, shape=(n_qs,n_pairs)
        The n_qs x n_pairs matrix that represents the accepting condition. If acc[q][i] is false then it means that q belongs to the first set of ith Rabin pair,
        if it is true, then q belongs to the second set and if it is none q doesn't belong either of them.
        
    spot_dra : spot.twa_graph
        The spot twa_graph object of the DRA.
        
    transition_probs : array, shape=(n_rows,n_cols,n_actions)
        The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.


    Parameters
    ----------
    shape : (n_rows, n_cols)
        The shape of the grid.
    
    structure : array, shape=(n_rows,n_cols)
        The structure of the grid function, structure[i][j] stores the type of the cell (i,j). 
        If structure[i,j] is 'E' it means the cell is empty and the agent is free to move in any direction. If it is 'B' then the cell is blocked, the agent cannot go there.
        If it is one of 'D','U','R' and 'L', the agent is free to enter the cell in any direction, but it cannot leave the cell in the opposite direction of the label.
            For example, if the label is 'D', then the agent cannot go up as if there is an obstacle there.
        If it is 'T', then the cell is a trap cell, which means if the agent cannot leave the cell once it reaches it.
        The default value is None.
        
    reward : array, shape = (n_rows, n_cols)
        The reward function, reward[i,j] is the reward for the state (i,j). If reward[i,j] is None, then the state is occupied by an obstacle.
        The default value is None.

    label : array, shape = (n_rows, n_cols)
        The labeling function, label[i,j] is the set of atomic propositions the state (i,j) is labeled with.
        The default value is None.
        
    A: list
        The list of actions represented by a string.
    
    p : float, optional
        The probability that the agent moves in the intended direction. It moves in one of the perpendicular direction with probability (1-p).
        The default value is 0.8.
    
    figsize: int, optional
        The size of the matplotlib figure to be drawn when the method plot is called. The default value is 5.
    
    cmap: matplotlib.colors.Colormap, optional
        The colormap to be used when drawing the plot of the MDP. The default value is matplotlib.cm.RdBu.
    
    """
    
    def __init__(self, shape, structure=None, reward=None, label=None, A=Actions, p=0.8, figsize=5, cmap=plt.cm.RdBu):
        self.shape = shape
        n_rows, n_cols = shape
        
        # Create the default structure, reward and label if they are not defined.
        self.structure = structure if structure is not None else np.full(shape,'E')
        self.reward = reward if reward is not None else np.zeros((n_rows,n_cols))
        self.label = label if label is not None else np.empty(shape,dtype=np.object); self.label.fill(()) if label is None else None
        
        self.p = p
        self.A = A
        
        # Create the transition matrix
        self.transition_probs = np.empty((n_rows, n_cols, len(A)),dtype=np.object)
        for state in self.states():
            for action, action_name in enumerate(A):
                self.transition_probs[state][action] = self.get_transition_prob(state,action_name)
        
        self.figsize = figsize
        self.cmap = cmap
        
    def states(self):
        """State generator.
        
        Yields
        ------
        state: tuple
            State coordinates (i,j).
        """
        n_rows, n_cols = self.shape
        for state in product(range(n_rows),range(n_cols)):
            yield state
        
    def random_state(self):
        """Generates a random state coordinate.
        
        Returns
        -------
        state: tuple
            A random state coordinate (i,j).
        """
        n_rows, n_cols = self.shape
        state = np.random.randint(n_rows),np.random.randint(n_cols)
        return state
                    
    def get_transition_prob(self,state,action_name):
        """Returns the list of possible next states with their probabilities when the action is taken (next_states,probs).
        The agent moves in the intented direction with a probability self.p; it can move sideways with a probability (1-self.p)/2. 
        If the direction is blocked by an obtacle or the agent is in a trap state then the agent stays in the same position.
    
        Parameters
        ----------
        state : tuple
            The coordinate of the state (i,j),
        
        action_name: str
            The name of the action.
        
        Returns
        -------
        out: (states,probs)
            The list of possible next states and their probabilities.
        """
        cell_type = self.structure[state]
        if cell_type in ['B', 'T']:
            return [state], np.array([1.])
        
        n_rows, n_cols = self.shape
        states, probs = [], []
        
        # South
        if action_name!='U' and state[0]+1 < n_rows and self.structure[state[0]+1][state[1]] != 'B' and cell_type != 'U':
            states.append((state[0]+1,state[1]))
            probs.append(self.p if action_name=='D' else (1-self.p)/2)
        # North
        if action_name!='D' and state[0]-1 >= 0 and self.structure[state[0]-1][state[1]] != 'B' and cell_type != 'D':
            states.append((state[0]-1,state[1]))
            probs.append(self.p if action_name=='U' else (1-self.p)/2) 
        # East
        if action_name!='L' and state[1]+1 < n_cols and self.structure[state[0]][state[1]+1] != 'B' and cell_type != 'L':
            states.append((state[0],state[1]+1))
            probs.append(self.p if action_name=='R' else (1-self.p)/2)
        # West
        if action_name!='R' and state[1]-1 >= 0 and self.structure[state[0]][state[1]-1] != 'B' and cell_type != 'R':
            states.append((state[0],state[1]-1))
            probs.append(self.p if action_name=='L' else (1-self.p)/2)
        
        # If the agent cannot move in some of the directions
        probs_sum = np.sum(probs)
        if probs_sum<1:
            states.append(state)
            probs.append(1-probs_sum)

        return states, probs
    
    def plot(self, value=None, policy=None, agent=None):
        """Plots the values of the states as a color matrix.
        
        Parameters
        ----------
        value : array, shape=(n_mdps,n_qs,n_rows,n_cols) 
            The value function. If it is None, the reward function will be plotted.
            
        policy : array, shape=(n_mdps,n_qs,n_rows,n_cols) 
            The policy to be visualized. It is optional.
            
        agent : tuple
            The position of the agent to be plotted. It is optional.

        """

        if value is None:
            value = self.reward

        # Dimensions
        n_rows, n_cols = self.shape
        
        # Plot
        fig = plt.figure(figsize=(self.figsize,self.figsize))
        threshold = np.nanmax(np.abs(value))
        threshold = 1 if threshold==0 else threshold 
        plt.imshow(value, interpolation='nearest', cmap=self.cmap, vmax=threshold, vmin=-threshold)
        
        # Get the axes
        ax = fig.axes[0]

        # Major ticks
        ax.set_xticks(np.arange(0, n_cols, 1))
        ax.set_yticks(np.arange(0, n_rows, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(n_cols+1))
        ax.set_yticklabels(np.arange(n_rows+1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
        
        # Move x axis to the top
        ax.xaxis.tick_top()

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        
        f=FontProperties(weight='bold')
        for i, j in self.states():  # For all states
            cell_type = self.structure[i,j]
            # If there is an obstacle
            if cell_type == 'B':
                circle=plt.Circle((j,i),0.5,color='darkgray')
                plt.gcf().gca().add_artist(circle)
            # If it is a trap cell
            elif cell_type == 'T':
                circle=plt.Circle((j,i),0.5,color='darkgray',fill=False)
                plt.gcf().gca().add_artist(circle)
                
            # If it is a directional cell (See the description of the class attribute 'structure' for details)
            elif cell_type == 'D':
                triangle = plt.Polygon([[j,i],[j-0.5,i-0.5],[j+0.5,i-0.5]], color='gray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'U':
                triangle = plt.Polygon([[j,i],[j-0.5,i+0.5],[j+0.5,i+0.5]], color='gray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'R':
                triangle = plt.Polygon([[j,i],[j-0.5,i+0.5],[j-0.5,i-0.5]], color='gray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'L':
                triangle = plt.Polygon([[j,i],[j+0.5,i+0.5],[j+0.5,i-0.5]], color='gray')
                plt.gca().add_patch(triangle)
            
            # If the background is too dark, make the text white
            color = 'white' if np.abs(value[i, j]) > threshold/2 else 'black'
            
            # Draw the arrows to visualize the policy
            if policy is not None:
                if policy[i,j] >= len(self.A):
                    plt.text(j, i, policy[i,j]-len(self.A), horizontalalignment='center',color=color)
                else:
                    action_name = self.A[policy[i,j]]
                    if action_name == 'D':
                        plt.arrow(j,i-.15,0,.2,head_width=.1,head_length=.1,color=color)
                    elif action_name == 'U':
                        plt.arrow(j,i+.15,0,-.2,head_width=.1,head_length=.1,color=color)
                    elif action_name == 'R':
                        plt.arrow(j-.15,i,.2,0,head_width=.1,head_length=.1,color=color)
                    elif action_name == 'L':
                        plt.arrow(j+.15,i,-.2,0,head_width=.1,head_length=.1,color=color)
    
            else:  # Print the values
                plt.text(j, i, format(value[i,j],'.2f'),horizontalalignment='center',color=color)  # Value
                plt.text(j, i+0.2, ','.join(self.label[i,j]),horizontalalignment='center',color=color,fontproperties=f)  
        
        if agent:  # Draw the agent
            circle=plt.Circle((agent[1],agent[0]),0.2,color='red')
            plt.gcf().gca().add_artist(circle)
        
    
    def plot_list(self,value_list,policy_list=None):
        """Plots the list of state values with a slider.
        
        Parameters
        ----------
        value_list : list of arrays with shape=(n_mdps,n_qs,n_rows,n_cols) 
            The list value functions.
            
        policy_list : list of arrays with  shape=(n_mdps,n_qs,n_rows,n_cols) 
            The policy to be visualized. It is optional.
        """
        # A helper function for the slider
        def plot_value(t):
            if policy_list is not None:
                self.plot(value_list[t],policy_list[t])
            else:
                self.plot(value_list[t])
            
        T = len(value_list)
        w=IntSlider(value=0,min=0,max=T-1)
        
        interact(plot_value,t=w)
