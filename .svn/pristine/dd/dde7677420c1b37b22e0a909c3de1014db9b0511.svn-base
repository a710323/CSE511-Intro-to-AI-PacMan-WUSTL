# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""
import util
import math
import time
import game
from game import Configuration


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    
    front = util.Stack();
    visited = [];
    front.push((problem.getStartState(), []));
    while not front.isEmpty():
        popState, popMoves = front.pop();
        if popState in visited:
            continue;
        if problem.isGoalState(popState):
            return popMoves;
        visited.append(popState);
        for state, direction, cost in problem.getSuccessors(popState):
            if state in visited:
                continue;
            front.push((state, popMoves+[direction]));


        
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    start = problem.getStartState();
    if problem.isGoalState(start):
        return [];
    frontier = util.Queue();
    frontier.push((start,[]));
    explored = set();
    while not frontier.isEmpty():
        popState, popMoves = frontier.pop();
        if popState in explored:
            continue;
        if problem.isGoalState(popState):
            return popMoves;
        explored.add(popState);
        successors = problem.getSuccessors(popState);
        for successor in successors:
            sucState, direction, cost = successor[0], successor[1], successor[2];
            if sucState not in explored:
                frontier.push((sucState, popMoves+[direction]));
    
def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    start = problem.getStartState();
    if problem.isGoalState(start):
        return[]
    frontier = util.PriorityQueue();
    frontier.push((start,[],0),0);
    explored = []
    while not frontier.isEmpty():
        popState, popMoves, popCost = frontier.pop();
        if popState in explored:
            continue;
        if problem.isGoalState(popState):
            return popMoves;
        explored.append(popState);
        for state, direction, cost in problem.getSuccessors(popState):
            if state in explored:
                continue;
            frontier.push((state, popMoves+[direction], popCost+cost), popCost+cost);
    return [];
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0;
    
def aStarSearch(problem, heuristic=util.manhattanDistance):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
##    heuristicCosts = 0;
##    start = problem.getStartState();
##    if problem.isGoalState(start):
##        return [];
##    frontier = util.PriorityQueue();
##    frontier.push((start,[],0,0),0);
##    explored = [];
##    while not frontier.isEmpty():
##        node = frontier.pop();
##        if problem.isGoalState(node[0]):
##            print "Total cost is " + str(node[2]);
##            print "Total heuristic is " + str(node[3]);
##            return (node[1]);
##        explored.append(node[0]);
##        for child in problem.getSuccessors(node[0]):
##            if (child[0] not in explored) and (child not in frontier.heap):
##                path = list(node[1]);
##                cost = child[2] + node [2];
##                heuristicCost = heuristic(child[0],problem);
##                path.append(child[1]);
##                f = cost + heuristicCost;
##                heuristicCosts = node[3] + heuristicCost;
##                frontier.push((child[0], path, cost, heuristicCosts),f)

    front = util.PriorityQueue();
    visited = set();
    front.push((problem.getStartState(),[],0),0);
    while not front.isEmpty():
        curState, curMoves, curCost = front.pop();
        if curState in visited:
            continue;
        if problem.isGoalState(curState):
            return curMoves;
        visited.add(curState);
        for state, direction, cost in problem.getSuccessors(curState):
            if state in visited:
                continue;
            hValue = heuristic(state, problem);
            front.push((state, curMoves+[direction],curCost+cost), curCost+cost+hValue);
    return [];
        

    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
