import sys
# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***" 
        # Get possible locations of ghost in next State
        ghosts = []
        for ghost in newGhostStates:
            gx, gy = ghost.configuration.getPosition()
            for i, j in [(0,0),(0,1),(1,0),(-1,0),(0,-1)]:
                ghosts.append((gx+i, gy+j))

        # Get new pacman position and food locations    
        current_food = currentGameState.getFood()
        x, y = newPos

        # return fail value when ghost catch pacman
        if newPos in ghosts:
            return -sys.maxsize
        # return reward for food when pacman eat food
        if current_food[x][y]:
            return 10

        # evaluate all food locations in the map by Manhantan distance, the farther the less points
        return 1/min([util.manhattanDistance(newPos, food) for food in current_food.asList()]) # max score is (w + h) which is maximum score can get from the Manhantan distance

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.MinimaxSearch(gameState, current_depth=1, agentIndex=0)
        
    def MinimaxSearch(self, gameState, current_depth, agentIndex):
        # stop when reach the depth or win or lose
        if current_depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # next possible moves
        next_moves = [action for action in gameState.getLegalActions(agentIndex) if action != 'Stop']
        
        # update next depth and next agent
        next_agent = agentIndex + 1 # pacman -> ghost_1 -> ghost_2 -> ...
        next_depth = current_depth
        if next_agent >= gameState.getNumAgents():
            next_agent = 0
            next_depth += 1
        
        # Evaluate next moves having maximum score
        moves_scores = [self.MinimaxSearch(gameState.generateSuccessor(agentIndex, action), next_depth, next_agent) for action in next_moves]
        
        # decide the next move here
        if agentIndex == 0 and current_depth == 1: # at root
            return next_moves[moves_scores.index(max(moves_scores))]
        
        # min for ghost, max for pacman
        if agentIndex:
            return min(moves_scores)
        return max(moves_scores)
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.AlphaBeta(gameState, current_depth=1, agentIndex=0, alpha=-sys.maxsize, beta=sys.maxsize, value = -sys.maxsize)
    
    def AlphaBeta(self, gameState, current_depth, agentIndex, alpha, beta, value):
        # stop when reach the depth or win or lose
        if current_depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), alpha, beta
        
        # next possible moves
        next_moves = [action for action in gameState.getLegalActions(agentIndex) if action != 'Stop']
        
        # update next depth and next agent
        next_agent = agentIndex + 1 # pacman -> ghost_1 -> ghost_2 -> ...
        next_depth = current_depth
        if next_agent >= gameState.getNumAgents():
            next_agent = 0
            next_depth += 1
        
        # decide the next move here
        if agentIndex == 0 and current_depth == 1: # at root always max node
            scores = []
            for action in next_moves:
                value, _, _ = self.AlphaBeta(gameState.generateSuccessor(agentIndex, action), next_depth, next_agent, alpha, beta, value)
                scores.append(value)
                alpha = max(scores) # update alpha from children
                if beta < value: # prune
                    break
            return next_moves[scores.index(max(scores))]
        
        # update alpha, beta
        if agentIndex: # min node
            value = sys.maxsize
            for action in next_moves:
                new_value, _, beta = self.AlphaBeta(gameState.generateSuccessor(agentIndex, action), next_depth, next_agent, alpha, beta, value)
                value = min(value, new_value) # compare new_value to old value, update if necessary
                if alpha > value: # prune
                    return value, alpha, beta
                beta = min(beta, value)
            return value, alpha, beta
        
        # max node
        value = -sys.maxsize
        for action in next_moves:
            new_value, alpha, _ = self.AlphaBeta(gameState.generateSuccessor(agentIndex, action), next_depth, next_agent, alpha, beta, value)
            value = max(value, new_value) # compare new_value to old value, update if necessary
            if beta < value: # prune
                return value, alpha, beta
            alpha = max(alpha, value)
        return value, alpha, beta

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
        return self.ExpectiMax(gameState, current_depth=1, agentIndex=0)

    def ExpectiMax(self, gameState, current_depth, agentIndex):
        # stop when reach the depth or win or lose
        if current_depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # next possible moves
        next_moves = [action for action in gameState.getLegalActions(agentIndex) if action != 'Stop']
                
        # update next depth and next agent
        next_agent = agentIndex + 1 # pacman -> ghost_1 -> ghost_2 -> ...
        next_depth = current_depth
        if next_agent >= gameState.getNumAgents():
            next_agent = 0
            next_depth += 1
        
        # Evaluate next moves having maximum score
        moves_scores = [self.ExpectiMax(gameState.generateSuccessor(agentIndex, action), next_depth, next_agent) for action in next_moves]
            
        # decide the next move here
        if agentIndex == 0 and current_depth == 1: # at root
            return next_moves[moves_scores.index(max(moves_scores))]
        
        if agentIndex: # expect node, calculate the mean
            return sum(moves_scores)/len(moves_scores)
        return max(moves_scores) # root node is max node

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    I present opposite relationship by the negative coefficient, the direct relationship by the positive coefficient
    The metric I collected as following
    1) Food: distance to closest food, number of food left
    2) Ghost: distance to closest ghost, distance to closest scared ghost
    3) Capsule: distance to closest capsule, number of capsule left
    4) Current score:
    """
    "*** YOUR CODE HERE ***"
    # win game
    if currentGameState.isWin():  
        return sys.maxsize
    # lose game
    if currentGameState.isLose():  
        return -sys.maxsize
    
    # setting up
    pacman = currentGameState.getPacmanPosition()

    # food
    foods = currentGameState.getFood().asList()
    food_closest = min([util.manhattanDistance(pacman, food) for food in foods])
    food_left = len(foods)

    # ghost
    ghosts = currentGameState.getGhostStates()
    ghost_active_list = [util.manhattanDistance(pacman, ghost.getPosition()) for ghost in ghosts if ghost.scaredTimer==0]
    ghost_active = min(ghost_active_list) if len(ghost_active_list) else 1
    ghost_scared_list = [util.manhattanDistance(pacman, ghost.getPosition()) for ghost in ghosts if ghost.scaredTimer]
    ghost_scared = min(ghost_scared_list) if len(ghost_scared_list) else 0

    # capsule
    capsules = currentGameState.getCapsules()
    capsule_closest = min([util.manhattanDistance(pacman, capsule) for capsule in capsules], default = 0)
    capsule_left = len(capsules)

    # score
    scores = currentGameState.getScore()

    scores = [-2*food_closest -4*food_left,
              -1/ghost_active - 2*ghost_scared,
              -10*capsule_closest - 20*capsule_left,
              100*scores]
    return sum(scores)

# Abbreviation
better = betterEvaluationFunction
