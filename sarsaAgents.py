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


class PacmanSarsaAgent(ReinforcementAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
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

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions is None or len(legalActions) == 0:
            return action
        maxQ = -1000
        for act in legalActions:
            q = self.getQValue(state, act)
            # print("action from q: < state - {} > = {}".format(act, q))
            if maxQ < q:
                action = act
                maxQ = q
        return action

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions is None or len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        self.doAction(state, action)
        return action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        nextAction = self.getAction(nextState)
        if nextAction is None:
            nextQVal = 0.
        else:
            nextQVal = self.getQValue(nextState, nextAction)
        difference = reward + self.discount * nextQVal - self.getQValue(state, action)
        # print("{0:.2f}(diff) = {1:.2f}(reward) + {2:.2f}(discount) x {3:.2f}(nextQVal) - {4:.2f}(currentQVal)".format(difference, reward, self.discount, nextQVal, self.getQValue(state, action)))
        features = self.featExtractor.getFeatures(state, action)
        for feature_key in features:
            if feature_key not in self.weights.keys():
                self.weights[feature_key] = random.uniform(-1.,1.)
            self.weights[feature_key] += self.alpha * difference * features[feature_key]

    def final(self, state):
        "Called at the end of each game."
        PacmanSarsaAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass
            # print(self.weights)
