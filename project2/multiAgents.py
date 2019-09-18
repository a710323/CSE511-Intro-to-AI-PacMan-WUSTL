# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    newFoodList = newFood.asList()
    min_food_distance = -1
    for food in newFoodList:
      distance = util.manhattanDistance(newPos, food)
      if min_food_distance >= distance or min_food_distance == -1:
        min_food_distance = distance

        """Calculating the distances from pacman to the ghosts. Also, checking for the proximity of the ghosts (at distance of 1) around pacman."""
    distances_to_ghosts = 1
    proximity_to_ghosts = 0
    for ghost_state in successorGameState.getGhostPositions():
      distance = util.manhattanDistance(newPos, ghost_state)
      distances_to_ghosts += distance
      if distance <= 1:
        proximity_to_ghosts += 1

        """Combination of the above calculated metrics."""
    return successorGameState.getScore() + (1 / float(min_food_distance)) - (1 / float(distances_to_ghosts)) - proximity_to_ghosts


  
def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
##  def maxValue(self, gameState, depth):
##    pacmanActions = gameState.getLegalActions(0)
##    if depth > self.depth or gameState.isWin() or not pacmanActions:
##      return self.evaluationFunction(gameState), None
##    actionCosts = []
##    for action in pacmanActions:
##      successor = gameState.generateSuccessor(0, action)
##      actionCosts.append((self.minValue(successor, 1, depth), action))
##    return max(actionCosts)
##
##  def minValue(self, gameState, agentIndex, depth):
##    ghostActions = gameState.getLegalActions(agentIndex)
##    if not ghostActions or gameState.isLose():
##      return self.evaluationFunction(gameState), None
##    if agentIndex == gameState.getNumAgents() - 1:
##      return min([self.maxValue(successor, depth + 1) for successor in [gameState.generateSuccessor(agentIndex, action) for action in ghostActions]])
##    else:
##      return min([self.minValue(successor, agentIndex + 1, depth) for successor in [gameState.generateSuccessor(agentIndex, action) for action in ghostActions]])
##
##  def getAction(self, gameState):
##    return self.maxValue(gameState, 1)[1]
    return self.minimax_value(gameState,0,0)
  
  def minimax_value(self, gameState, Index, nodeDepth):

      if Index >= gameState.getNumAgents():
          Index = 0
          nodeDepth += 1

      if nodeDepth == self.depth:
          return self.evaluationFunction(gameState)

      if Index == self.index:
          return self.max_value(gameState, Index, nodeDepth)
      else:
          return self.min_value(gameState, Index, nodeDepth)

      return 'None'
   
  def max_value(self, gameState, Index, nodeDepth):

    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    value = float("-inf")
    actionValue = "None"

    legalActions = gameState.getLegalActions(Index)

    for actions in legalActions:

      if actions == Directions.STOP:
        continue;

      successor = gameState.generateSuccessor(Index,actions)
      temp = self.minimax_value(successor,Index+1,nodeDepth)

      if temp > value:
        value = max(temp,value)
        actionValue = actions

    if nodeDepth == 0:
      return actionValue
    else:
      return value

  def min_value(self, gameState, Index, nodeDepth):

    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    value = float("inf")
    actionValue = "None"

    legalActions = gameState.getLegalActions(Index)
    agentNumber = gameState.getNumAgents()

    for actions in legalActions:

      if actions == Directions.STOP:
        continue;

      successor = gameState.generateSuccessor(Index,actions)
      temp = self.minimax_value(successor,Index+1,nodeDepth)

      if temp < value:
        value = min(temp,value)
        actionValue = actions


    return value




class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    scores = []
    alpha = -float("inf");
    beta = float("inf");
    for action in gameState.getLegalActions(0):
      newState = gameState.generateSuccessor(0, action)
      score = self.minValue(newState, self.depth, 1, alpha, beta)
      scores.append((score, action))
      if max(scores)[0] > beta:
        return max(scores)[1]
      alpha = max(max(scores)[0], alpha)
    return max(scores)[1]


  def maxValue(self, gameState, currentDepth, alpha, beta):
    legalMoves = gameState.getLegalActions(0)
    if gameState.isWin() or gameState.isLose() or currentDepth < 1:
      return self.evaluationFunction(gameState)
    scores = []
    for action in legalMoves:
      scores.append(self.minValue(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta))
      maximum = max(scores)
      if maximum > beta:
        return maximum
      alpha = max(maximum, alpha)

    if scores:
      return max(scores)
    else:
      return -inf

  def minValue(self, gameState, currentDepth, ghostIndex, alpha, beta):
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)
    scores = []
    legalMoves = gameState.getLegalActions(ghostIndex)
    for action in legalMoves:
      ghostState = gameState.generateSuccessor(ghostIndex, action)
      if ghostIndex == gameState.getNumAgents() - 1:
        newVal = self.maxValue(ghostState, currentDepth - 1, alpha, beta)
      else:
        newVal = self.minValue(ghostState, currentDepth, ghostIndex + 1, alpha, beta)
      scores.append(newVal)
      minimum = min(scores)
      if minimum < alpha:
        return minimum
      beta = min(beta, minimum)
    if scores:
      return minimum
    else:
      return inf

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"

    moves = gameState.getLegalActions(0)
    scores = [self.expMinValue(gameState.generateSuccessor(0, action), self.depth, 1) for action in moves]

    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    return moves[chosenIndex]



  def expMaxValue(self, gameState, currentDepth):
    agentIndex = 0
    legalMoves = gameState.getLegalActions(0)
    if gameState.isWin() or gameState.isLose() or currentDepth < 1:
      return self.evaluationFunction(gameState)
    scores = [self.expMinValue(gameState.generateSuccessor(agentIndex, action), currentDepth, 1) for action in legalMoves]
    if scores:
      return max(scores)
    else:
      return -inf

  def expMinValue(self, gameState, currentDepth, ghostIndex):
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)
    ghostMoves = gameState.getLegalActions(ghostIndex)
    scores = []
    for action in ghostMoves:
      ghostState = gameState.generateSuccessor(ghostIndex, action)
      if ghostIndex == gameState.getNumAgents() - 1:
        expVal = self.expMaxValue(ghostState, currentDepth - 1)
      else:
        expVal = self.expMinValue(ghostState, currentDepth, ghostIndex + 1)
      scores.append(expVal)

    if scores:
      return sum(scores)/len(ghostMoves)
    else:
      return inf



def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  if currentGameState.isWin():
      return float("inf")
  elif currentGameState.isLose():
      return float("-inf")

  foodPositions = currentGameState.getFood().asList()
  pacmanState = currentGameState.getPacmanState()
  pacmanPosition = pacmanState.getPosition()
  foodCount = currentGameState.getNumFood()

  if foodCount == 0:
      return float("inf")

  score = 0
  ghostStates = currentGameState.getGhostStates()
  ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  ghostPositions = [ghostState.configuration.pos for ghostState in ghostStates]
  ghostDistance = [manhattanDistance(pacmanPosition, ghostPosition) for ghostPosition in ghostPositions]

  minGhostDistance = 10e5
  for i in range(0, len(ghostScaredTimes)):
      minGhostDistance = min(minGhostDistance, ghostDistance[i])
      if ghostScaredTimes[i] <= 1 and ghostDistance[i] <= 1:
          return float("-inf")
      else:
          score += ghostScaredTimes[i] * 10
          if ghostDistance[i] == 0:
              score += 200

  if minGhostDistance <= 1:
      return float("-inf")

  score -= currentGameState.getNumFood() * 30
  score -= len(currentGameState.getCapsules()) * 200

  while foodPositions:
      closestPosition = min(foodPositions, key = lambda x: manhattanDistance(pacmanPosition, x))
      actual_distance = manhattanDistance( pacmanPosition, closestPosition)
      score -= actual_distance
      foodPositions.remove(closestPosition)
      pacmanPosition = closestPosition

  return score


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

