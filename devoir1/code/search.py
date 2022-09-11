# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import queue
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    from util import Stack

    stack = Stack()
    current_path = []
    current_state = problem.getStartState()
    seen_states = []
    stack.push((current_state, current_path))
    while not stack.isEmpty():
        current_state, current_path = stack.pop()
        if problem.isGoalState(current_state):
            return current_path
        seen_states.append(current_state)
        for successor in problem.getSuccessors(current_state):
            if successor[0] not in seen_states:
                stack.push((successor[0], current_path + [successor[1]]))
        

    return None
"""
    from util import Stack


    stack = Stack()
    current_path = []
    
    current_state = problem.getStartState()
    seen_states = []
    while not problem.isGoalState(current_state):
        
        seen_states.append(current_state)
        for successor in problem.getSuccessors(current_state):
            if successor[0] not in seen_states:
                stack.push((successor[0], current_path + [successor[1]]))
        current_state, current_path = stack.pop()

    return current_path

"""



    


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    queue = util.Queue()
    current_path = []
    current_state = problem.getStartState()
    seen_states = [problem.getStartState()]
    while not problem.isGoalState(current_state):
        for successor in problem.getSuccessors(current_state):
            if successor[0] not in seen_states:
                queue.push((successor[0], current_path + [successor[1]]))
                seen_states.append(successor[0])
        current_state, current_path = queue.pop()

    return  current_path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priority_queue = util.PriorityQueue()
    current_path = []
    current_state = problem.getStartState()
    seen_states = []
    current_cost = 0
    while not problem.isGoalState(current_state):
        
        if(current_state not in seen_states):
            seen_states.append(current_state)
            for successor in problem.getSuccessors(current_state):
                if successor[0] not in seen_states:

                    priority_queue.update((successor[0], current_path + [successor[1]],current_cost+successor[2]), current_cost+successor[2])
                
        current_state, current_path, current_cost = priority_queue.pop()


    return  current_path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priority_queue = util.PriorityQueue()
    current_path = []
    current_state = problem.getStartState()
    seen_states = []
    current_cost = 0
    while not problem.isGoalState(current_state):
        
        if(current_state not in seen_states):
            seen_states.append(current_state)
            for successor in problem.getSuccessors(current_state):
                if successor[0] not in seen_states:
                    priority_queue.update((successor[0], current_path + [successor[1]],current_cost+successor[2]), current_cost+successor[2]+heuristic(successor[0],problem))
                
        current_state, current_path, current_cost = priority_queue.pop()


    return  current_path



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
