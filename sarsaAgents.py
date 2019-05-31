# https://github.com/takeitallsource/berkeley-cs-188/blob/master/project-3/reinforcement/SarsaAgents.py

# SarsaAgents.py
# ------------------
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


import random
import util

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

def createApproximateSarsaAgent(num_pacmen, agent, start_index, **args):
    return [eval(agent)(index=i, **args) for i in range(start_index, start_index + num_pacmen)]

class PacmanSarsaAgent(ReinforcementAgent):
    def __init__(self, index, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = index
        self.isDead = False
        self.hasStart = False
        ReinforcementAgent.__init__(self, **args)


class ApproximateSarsaAgent(PacmanSarsaAgent):
    def __init__(self, extractor='ComplexExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        self.weights = util.Counter()
        PacmanSarsaAgent.__init__(self, **args)

    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if legalActions is None:
            return 0.
        return max([self.getQValue(state, action) for action in legalActions])

    def computeActionFromQValues(self, state, total_pacmen, agentIndex):
        legalActions = self.getLegalActions(state, total_pacmen, agentIndex)
        action = None
        if legalActions is None or len(legalActions) == 0:
            return action
        maxQ = -10000000
        for act in legalActions:
            q = self.getQValue(state, act, agentIndex, total_pacmen)
            # print("action from q: < state - {} > = {}".format(act, q))
            if maxQ < q:
                action = act
                maxQ = q
        return action

    def getAction(self, state, total_pacmen, agentIndex):
        legalActions = self.getLegalActions(state, total_pacmen, agentIndex)
        action = None
        if legalActions is None or len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state, total_pacmen, agentIndex)
        self.doAction(state, action)
        return action

    def getPolicy(self, state, total_pacmen, agentIndex):
        return self.computeActionFromQValues(state, total_pacmen, agentIndex)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action, agentIndex, total_pacmen):
        return self.weights * self.featExtractor.getFeatures(state, action, agentIndex, total_pacmen)

    def update(self, state, action, nextState, reward, total_pacmen, agentIndex):
        nextAction = self.getAction(nextState, total_pacmen, agentIndex)
        if nextAction is None:
            nextQVal = 0.
        else:
            nextQVal = self.getQValue(nextState, nextAction, agentIndex, total_pacmen)
        difference = reward + self.discount * nextQVal - self.getQValue(state, action, agentIndex, total_pacmen)
        # print("{0:.2f}(diff) = {1:.2f}(reward) + {2:.2f}(discount) x {3:.2f}(nextQVal) - {4:.2f}(currentQVal)".format(difference, reward, self.discount, nextQVal, self.getQValue(state, action)))
        features = self.featExtractor.getFeatures(state, action, agentIndex, total_pacmen)
        for feature_key in features:
            if feature_key not in self.weights.keys():
                self.weights[feature_key] = random.uniform(-1.,1.)
            self.weights[feature_key] += self.alpha * difference * features[feature_key]

    def final(self, state, total_pacmen, agentIndex):
        "Called at the end of each game."
        PacmanSarsaAgent.final(self, state, total_pacmen, agentIndex)
        if self.episodesSoFar == self.numTraining:
            pass
            # print(self.weights)
