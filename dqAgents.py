import os
import util
import numpy as np

from dqn import DQN
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import torch
import torch.cuda
import torch.optim as optim

def createPacmanDQAgent(num_pacmen, agent, start_index, **args):
    return [eval(agent)(index=i, **args) for i in range(start_index, start_index + num_pacmen)]

class DQAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)

class PacmanDQAgent(DQAgent):
    def __init__(self, index, extractor='ComplexExtractor', **args):
        if os.path.exists("model_param/dqn.pt"):
            self.dqnet = torch.load("model_param/dqn.pt")
        else:
            self.dqnet = DQN(n_features=8)
        if torch.cuda.is_available():
            self.dqnet.cuda()
        self.opti = optim.Adam(self.dqnet.parameters(), lr=.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.opti, 10000, gamma=.1)
        self.feat_extractor = util.lookup(extractor, globals())()
        self.action_mapping = {'North':0, 'South':1, 'East':2, 'West':3, 'Stop':4}
        self.replay_buffer = []
        self.index = index
        self.isDead = False
        DQAgent.__init__(self, **args)
    
    def getFeature(self, state, action, total_pacmen, agentIndex):
        feature_dict = self.feat_extractor.getFeatures(state, action, total_pacmen, agentIndex)
        feature_dict["action"] = self.action_mapping[action]
        feature = DQN.dict2vec(feature_dict)
        return feature

    def getQValue(self, state, action, total_pacmen, agentIndex):
        if torch.cuda.is_available():
            feature = torch.from_numpy(self.getFeature(state, action, total_pacmen, agentIndex)).cuda()
        else:
            feature = torch.from_numpy(self.getFeature(state, action, total_pacmen, agentIndex))
        return self.dqnet(feature).detach().cpu().numpy()[0]
    
    def computeValueFromQValues(self, state, total_pacmen, agentIndex):
        legalActions = self.getLegalActions(state, total_pacmen, agentIndex)
        if legalActions is None or len(legalActions) == 0:
            return 0.
        return max([self.getQValue(state, action, total_pacmen, agentIndex) for action in legalActions])
    
    def computeActionFromQValues(self, state, total_pacmen, agentIndex):
        legalActions = self.getLegalActions(state, total_pacmen, agentIndex)
        action = None
        if legalActions is None or len(legalActions) == 0:
            return action
        maxQ = -1000000
        for act in legalActions:
            q = self.getQValue(state, act, total_pacmen, agentIndex)
            if maxQ < q:
                action = act
                maxQ = q
        return action
    
    def getAction(self, state, total_pacmen, agentIndex):
        legalActions = self.getLegalActions(state, total_pacmen, agentIndex)
        action = None
        if legalActions is None or len(legalActions) == 0:
            return action
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
    
    def update(self, state, action, nextState, reward, total_pacmen, agentIndex):
        feature = self.getFeature(state, action, total_pacmen, agentIndex)
        target = reward + self.discount * self.computeValueFromQValues(nextState, total_pacmen, agentIndex)
        self.store_trajectory(feature, target)
        if self.episodesSoFar < self.numTraining:
            self.replay()
        else:
            self.dqnet.eval()
    
    def store_trajectory(self, feature, target):
        self.replay_buffer.append((feature, target))
    
    def replay(self):
        n_samples = len(self.replay_buffer)
        x = np.array([pairs[0] for pairs in self.replay_buffer[-min(1000, n_samples):]]).astype(np.float32)
        y = np.array([pairs[1] for pairs in self.replay_buffer[-min(1000, n_samples):]]).astype(np.float32)[:,np.newaxis]
        self.train(x, y)
    
    def train(self, x, y):
        batch_size = 32
        epochs = 10
        n_samples = x.shape[0]
        if n_samples < 1:
            return
        if n_samples < batch_size:
            batch_size = n_samples
        batches = math.ceil(1.*n_samples/batch_size)
        random_index = np.arange(n_samples)
        avg_loss = 0.
        for _ in range(epochs):
            np.random.shuffle(random_index)
            for i in range(int(batches)):
                start_index = i*batch_size
                end_index = min(start_index+batch_size, n_samples)
                if torch.cuda.is_available():
                    batch_x = torch.from_numpy(x[start_index:end_index]).cuda()
                else:
                    batch_x = torch.from_numpy(x[start_index:end_index])
                batch_y = y[start_index:end_index] / 100     # ...
                outp = self.dqnet(batch_x)
                if torch.cuda.is_available():
                    loss = self.dqnet.criterion(outp, torch.from_numpy(batch_y).cuda())
                else:
                    loss = self.dqnet.criterion(outp, torch.from_numpy(batch_y))
                self.opti.zero_grad()
                loss.backward()
                self.opti.step()
                avg_loss += loss.detach().cpu().numpy()
            self.scheduler.step()
        print(">>> agent {} - average loss {} - lr {}".format(self.index, avg_loss/batches/epochs, self.scheduler.get_lr()))

    def final(self, state, total_pacmen, agentIndex):
        "Called at the end of each game."
        # call the super-class final method
        DQAgent.final(self, state, total_pacmen, agentIndex)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            torch.save(self.dqnet, "model_param/dqn.pt")
            print("DQN model saved at 'model_param/dqn.pt'")
