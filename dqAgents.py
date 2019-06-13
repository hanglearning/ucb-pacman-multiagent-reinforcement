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

inf = 1000000

def createPacmanDQAgent(num_pacmen, agent, start_index, **args):
    return [eval(agent)(index=i, **args) for i in range(start_index, start_index + num_pacmen)]

class DQAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)

class PacmanDQAgent(DQAgent):
    def __init__(self, index, **args):
        if os.path.exists(f"model_param/dqn_{index}.pt"):
            self.dqnet = torch.load(f"model_param/dqn_{index}.pt")
        else:
            self.dqnet = DQN(n_features=8)
        if torch.cuda.is_available():
            self.dqnet.cuda()
        self.opti = optim.Adam(self.dqnet.parameters(), lr=.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.opti, 10000, gamma=.1)
        self.feat_extractor = util.lookup('ComplexExtractor', globals())()
        self.action_mapping = {'North':0, 'South':1, 'East':2, 'West':3, 'Stop':4}
        self.action_mapping_reverse = {0:'North', 1:'South', 2:'East', 3: 'West', 4: 'Stop'}
        self.replay_buffer = []
        self.index = index
        self.isDead = False
        self.hasStarted = False
        self.isPacman = True
        self.hasFinishedTraining = False
        self.scoreChange = 0
        DQAgent.__init__(self, **args)
    
    def getFeature(self, state, total_pacmen, agentIndex):
        feature_dict = self.feat_extractor.getFeatures(state, total_pacmen, agentIndex)
        feature = DQN.dict2vec(feature_dict)
        return feature

    def getQValue(self, state, total_pacmen, agentIndex):
        feature = torch.from_numpy(self.getFeature(state, total_pacmen, agentIndex))
        if torch.cuda.is_available():
            feature = feature.cuda()
        return self.dqnet(feature).cpu().detach().numpy()
    
    def computeActionValueFromQValues(self, state, total_pacmen, agentIndex):
        legalActions = [self.action_mapping[act] for act in self.getLegalActions(state, total_pacmen, agentIndex)]
        if legalActions is None or len(legalActions) == 0:
            return None, 0.
        q_values = self.getQValue(state, total_pacmen, agentIndex)
        action_q = -inf
        for i in range(5):
            if i not in legalActions:
                continue
            action = self.action_mapping_reverse[i]
            action_q = q_values[i]
        return action, action_q
    
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
        return self.computeActionValueFromQValues(state, total_pacmen, agentIndex)[0]
    
    def getValue(self, state, total_pacmen, agentIndex):
        return self.computeActionValueFromQValues(state, total_pacmen, agentIndex)[1]
    
    def update(self, state, action, nextState, reward, total_pacmen, agentIndex, stillTraining):
        feature = self.getFeature(state, total_pacmen, agentIndex)
        target = reward + self.discount * self.computeActionValueFromQValues(nextState, total_pacmen, agentIndex)[1]
        self.store_trajectory(feature, target, self.action_mapping[action])
        # if self.episodesSoFar < self.numTraining:
        if stillTraining:
            self.replay()
        else:
            self.dqnet.eval()
    
    def store_trajectory(self, feature, target, action):
        self.replay_buffer.append((feature, target, action))
    
    def replay(self):
        n_samples = len(self.replay_buffer)
        x = np.array([pairs[0] for pairs in self.replay_buffer[-min(1000, n_samples):]]).astype(np.float32)
        y = np.array([pairs[1] for pairs in self.replay_buffer[-min(1000, n_samples):]]).astype(np.float32)[:,np.newaxis]
        actions = np.array([pairs[2] for pairs in self.replay_buffer[-min(1000, n_samples):]]).astype(np.long)
        self.train(x, y, actions)
    
    def train(self, x, y, actions):
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
                batch_x = torch.from_numpy(x[start_index:end_index])
                batch_y = torch.from_numpy(y[start_index:end_index])     # ...
                batch_action = actions[start_index:end_index]
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                outp = self.dqnet(batch_x)
                outp = outp[range(end_index-start_index), batch_action].view(-1,1)
                loss = self.dqnet.criterion(outp, batch_y)
                self.opti.zero_grad()
                loss.backward()
                self.opti.step()
                avg_loss += loss.detach().cpu().numpy()
            self.scheduler.step()
        # print(">>> agent {} - average loss {} - lr {}".format(self.index, avg_loss/batches/epochs, self.scheduler.get_lr()))

    def final(self, state, total_pacmen, agentIndex, stillTraining, forceFinish):
        "Called at the end of each game."
        # call the super-class final method
        DQAgent.final(self, state, total_pacmen, agentIndex, stillTraining, forceFinish)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            if not os.path.exists('model_param'):
                os.makedirs('model_param')
            torch.save(self.dqnet, f"model_param/dqn_{agentIndex}.pt")
            print(f"DQN model for pacman {agentIndex} saved at 'model_param/dqn_{agentIndex}.pt'")
