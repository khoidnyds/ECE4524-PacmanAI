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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

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
            return -1
        # return reward for food when pacman eat food
        if current_food[x][y]:
            return sys.maxsize

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
        next_moves = [action for action in gameState.getLegalActions(agentIndex) if action is not 'Stop']
        
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
        next_moves = [action for action in gameState.getLegalActions(agentIndex) if action is not 'Stop']
        
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
        util.raiseNotDefined()

    def ExpectiMax(self, gameState, current_depth, agentIndex):
        # stop when reach the depth or win or lose
        if current_depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # next possible moves
        next_moves = [action for action in gameState.getLegalActions(agentIndex) if action is not 'Stop']
                
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

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin() :  return sys.maxsize
    if currentGameState.isLose() :  return -sys.maxsize
    
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    capsulePos = currentGameState.getCapsules()
    
    weightFood, weightGhost, weightCapsule, weightHunter = 5.0, 5.0, 5.0, 0.0
    ghostScore, capsuleScore, hunterScore = 0.0, 0.0, 0.0
    #print '\nnewPos '+str(newPos)
    #print 'oldFood '+str(oldFood.asList())
    #print "score %d" % successorGameState.getScore()
    
    "obtain food score" # may closestFood be zero?
    currentFoodList = currentFood.asList()
    closestFood = min([util.manhattanDistance(currentPos, foodPos) for foodPos in currentFoodList])
    foodScore = 1.0 / closestFood
    
    if GhostStates:
        ghostPositions = [ghostState.getPosition() for ghostState in GhostStates]
        ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
        ghostDistances = [util.manhattanDistance(currentPos, ghostPos) for ghostPos in ghostPositions]
        
        if sum(ScaredTimes) == 0 : 
            closestGhost = min(ghostDistances)
            ghostCenterPos = ( sum([ghostPos[0] for ghostPos in ghostPositions])/len(GhostStates),\
                               sum([ghostPos[1] for ghostPos in ghostPositions])/len(GhostStates))
            ghostCenterDist = util.manhattanDistance(currentPos, ghostCenterPos)
            if ghostCenterDist <= closestGhost and closestGhost >= 1 and closestGhost <= 5:
                if len(capsulePos) != 0:
                    closestCapsule = min([util.manhattanDistance(capsule,currentPos) for capsule in capsulePos])
                    if closestCapsule <= 3:
                        weightCapsule, capsuleScore = 20.0, (1.0 / closestCapsule)
                        weightGhost, ghostScore = 3.0, (-1.0 / (ghostCenterDist+1))
                    else:
                        weightGhost, ghostScore = 10.0, (-1.0 / (ghostCenterDist+1))
                else:
                    weightGhost, ghostScore = 10.0, (-1.0 / (ghostCenterDist+1))
            elif ghostCenterDist >= closestGhost and closestGhost >= 1 :
                weightFood *= 2
                if len(capsulePos) != 0:
                    closestCapsule = min([util.manhattanDistance(capsule,currentPos) for capsule in capsulePos])
                    if closestCapsule <= 3:
                        weightCapsule, capsuleScore = 15.0, (1.0 / closestCapsule)
                        weightGhost, ghostScore = 3.0, (-1.0 / closestGhost)
                    else:
                        ghostScore = -1.0 / closestGhost
                else:
                    ghostScore = -1.0 / closestGhost
            elif closestGhost == 0:
                return -sys.maxsize
            elif closestGhost == 1:
                weightGhost, ghostScore = 15.0, (-1.0 / closestGhost)
            else:
                ghostScore = -1.0 / closestGhost
        else:
            normalGhostDist = []
            closestPrey = sys.maxsize
            ghostCenterX, ghostCenterY = 0.0, 0.0
            for (index, ghostDist) in enumerate(ghostDistances):
                if ScaredTimes[index] == 0 :
                    normalGhostDist.append(ghostDist)
                    ghostCenterX += ghostPositions[index][0]
                    ghostCenterY += ghostPositions[index][1]
                else:
                    if ghostDist <= ScaredTimes[index] :
                        if ghostDist < closestPrey:
                            closestPrey = ghostDistances[index]
            if normalGhostDist:
                closestGhost = min(normalGhostDist)
                ghostCenterPos = ( ghostCenterX/len(normalGhostDist), ghostCenterY/len(normalGhostDist))
                ghostCenterDist = util.manhattanDistance(currentPos, ghostCenterPos)
                if ghostCenterDist <= closestGhost and closestGhost >= 1 and closestGhost <= 5:
                    weightGhost, ghostScore = 10.0, (- 1.0 / (ghostCenterDist+1))
                elif ghostCenterDist >= closestGhost and closestGhost >= 1 :
                    ghostScore = -1.0 / closestGhost
                elif closestGhost == 0:
                    return -sys.maxsize
                elif closestGhost == 1:
                    weightGhost, ghostScore = 15.0, (-1.0 / closestGhost)
                else:
                    ghostScore = - 1.0 / closestGhost
            weightHunter, hunterScore = 35.0, (1.0 / closestPrey)
    
    heuristic = currentGameState.getScore() + \
                weightFood*foodScore + weightGhost*ghostScore + \
                weightCapsule*capsuleScore + weightHunter*hunterScore
    return heuristic

# Abbreviation
better = betterEvaluationFunction
