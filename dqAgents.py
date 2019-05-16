import util
import numpy as np

from dqn import DQN
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import torch
import torch.optim as optim

class DQAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)

class PacmanDQAgent(DQAgent):
    def __init__(self, extractor='ComplexExtractor', **args):
        self.dqnet = DQN(n_features=8)
        self.dqnet.cuda()
        self.feat_extractor = util.lookup(extractor, globals())()
        self.action_mapping = {'North':0, 'South':1, 'East':2, 'West':3, 'Stop':4}
        self.replay_buffer = []
        self.index = 0
        DQAgent.__init__(self, **args)
    
    def getFeature(self, state, action):
        feature_dict = self.feat_extractor.getFeatures(state, action)
        feature_dict["action"] = self.action_mapping[action]
        feature = DQN.dict2vec(feature_dict)
        return feature

    def getQValue(self, state, action):
        feature = torch.from_numpy(self.getFeature(state, action)).cuda()
        return self.dqnet(feature).detach().cpu().numpy()[0]
    
    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if legalActions is None or len(legalActions) == 0:
            return 0.
        return max([self.getQValue(state, action) for action in legalActions])
    
    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions is None or len(legalActions) == 0:
            return action
        maxQ = -1000000
        for act in legalActions:
            q = self.getQValue(state, act)
            if maxQ < q:
                action = act
                maxQ = q
        return action
    
    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions is None or len(legalActions) == 0:
            return action
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
    
    def update(self, state, action, nextState, reward):
        feature = self.getFeature(state, action)
        target = reward + self.discount * self.computeValueFromQValues(nextState)
        self.store_trajectory(feature, target)
        self.replay()
    
    def store_trajectory(self, feature, target):
        self.replay_buffer.append((feature, target))
    
    def replay(self):
        n_samples = len(self.replay_buffer)
        x = np.array([pairs[0] for pairs in self.replay_buffer[-min(10000, n_samples):]]).astype(np.float32)
        y = np.array([pairs[1] for pairs in self.replay_buffer[-min(10000, n_samples):]]).astype(np.float32)[:,np.newaxis]
        self.train(x, y)
    
    def train(self, x, y, lr=1e-2, batch_size=20, episode=10):
        opti = optim.Adam(self.dqnet.parameters(), lr=lr)
        n_samples = x.shape[0]
        if n_samples < 1:
            return
        if n_samples < batch_size:
            batch_size = n_samples
        batches = math.ceil(1.*n_samples/batch_size)
        random_index = np.arange(n_samples)
        avg_loss = 0.
        for ep in range(episode):
            np.random.shuffle(random_index)
            for i in range(int(batches)):
                start_index = i*batch_size
                end_index = min(start_index+batch_size, n_samples)
                batch_x = torch.from_numpy(x[start_index:end_index]).cuda()
                batch_y = y[start_index:end_index]
                outp = self.dqnet(batch_x)
                loss = self.dqnet.criterion(outp, torch.from_numpy(batch_y).cuda())
                # print(loss.detach().numpy())
                # print("-----")
                # print("grad: {}".format(self.dqnet.fc1.weight.detach().cpu().numpy()))
                # print("-----")
                opti.zero_grad()
                loss.backward()
                opti.step()
                avg_loss += loss.detach().cpu().numpy()
        print(">>> average loss: {}".format(avg_loss/batches/episode))

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        DQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # print(self.weights)
