# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

import numpy as np
from game import Directions, Actions
import util, math

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return (pos_x, pos_y, dist)
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def closestCapsule(pos, capsule, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if (pos_x, pos_y) in capsule:
            return (pos_x, pos_y, dist)
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    return None

# class SimpleExtractor(FeatureExtractor):
#     """
#     Returns simple features for a basic reflex Pacman:
#     - whether food will be eaten
#     - how far away the next food is
#     - whether a ghost collision is imminent
#     - whether a ghost is one step away
#     """

#     def getFeatures(self, state, action):
#         # extract the grid of food and wall locations and get the ghost locations
#         food = state.getFood()
#         walls = state.getWalls()
#         ghosts = state.getGhostPositions()

#         features = util.Counter()

#         features["bias"] = 1.0

#         # compute the location of pacman after he takes the action
#         x, y = state.getPacmanPosition()
#         dx, dy = Actions.directionToVector(action)
#         next_x, next_y = int(x + dx), int(y + dy)

#         # count the number of ghosts 1-step away
#         features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

#         # if there is no danger of ghosts then add the food feature
#         if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
#             features["eats-food"] = 1.0

#         dist = closestFood((next_x, next_y), food, walls)
#         if dist is not None:
#             # make the distance a number less than one otherwise the update
#             # will diverge wildly
#             features["closest-food"] = float(dist) / (walls.width * walls.height)
#         features.divideAll(10.0)
#         return features

class ComplexExtractor(FeatureExtractor):
    def getFeatures(self, state, action, agentIndex, total_pacmen):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions(total_pacmen)
        ghosts_state = state.getGhostStates(total_pacmen)

        features = util.Counter()

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition(agentIndex)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        features["#-of-ghosts-1-step-away-edible"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for i, g in enumerate(ghosts) if ghosts_state[i].scaredTimer > 0)

        # if there is no danger of ghosts then add the food feature
        features["eats-food"] = 0.
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.
        
        dist_ghost = list(map(lambda pos_g: abs(pos_g[0]-next_x)+abs(pos_g[1]-next_y), ghosts[:-1]))
        nearest = walls.width * walls.height
        features["nearest-ghost-edible"] = 0.
        for i, d in enumerate(dist_ghost):
            if d < nearest:
                nearest = d
                if ghosts_state[i].scaredTimer > 0:
                    features["nearest-ghost-edible"] = 10.
        features["closest-ghost"] = nearest / (walls.width * walls.height)

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist[2]) / (walls.width * walls.height)

        features.divideAll(10.0)

        features["bias"] = 1.0

        return features

class ComplexExtractor_DQN(FeatureExtractor):
    def getFeatures(self, state, agentIndex, total_pacmen):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        capsule = state.getCapsules()
        walls = state.getWalls()
        ghosts = state.getGhostPositions(total_pacmen)
        ghosts_state = state.getGhostStates(total_pacmen)

        features = util.Counter()

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition(agentIndex)
        x, y = int(x), int(y)

        dist_ghost = list(map(lambda pos_g: abs(pos_g[0]-x)+abs(pos_g[1]-y), ghosts[:-1]))
        features["#-of-ghosts-2-step-away"] = 0
        features["#-of-ghosts-5-step-away"] = 0
        for i, d in enumerate(dist_ghost):
            if ghosts_state[i].scaredTimer > 0:
                continue
            if d == 2:
                features["#-of-ghosts-2-step-away"] += 1
            elif d <= 5:
                features["#-of-ghosts-5-step-away"] += 1
        
        features.divideAll(10.0)

        features["closest-ghost"] = min(dist_ghost)/(walls.width + walls.height)
        closest_ghost = ghosts[np.argmax(dist_ghost)]
        features["closest-ghost-status"] = 1. if ghosts_state[np.argmax(dist_ghost)].scaredTimer > 0 else 0.
        if closest_ghost[0] >= x and closest_ghost[1] >= y:
            features["closest-ghost-direction"] = 1
        elif closest_ghost[0] < x and closest_ghost[1] >= y:
            features["closest-ghost-direction"] = 2
        elif closest_ghost[0] < x and closest_ghost[1] < y:
            features["closest-ghost-direction"] = 3
        else:
            features["closest-ghost-direction"] = 4

        features["closest-food"] = 1.
        dist_food = None
        closest_food = closestFood((x, y), food, walls)
        if closest_food is None:
            features["closest-food-direction"] = 0
        elif closest_food[0] >= x and closest_food[1] >= y:
            features["closest-food-direction"] = 1
        elif closest_food[0] < x and closest_food[1] >= y:
            features["closest-food-direction"] = 2
        elif closest_food[0] < x and closest_food[1] < y:
            features["closest-food-direction"] = 3
        else:
            features["closest-food-direction"] = 4
        if closest_food is not None:
            dist_food = abs(closest_food[0]-x)+abs(closest_food[1]-y)
        
        features["closest-capsule"] = 1.
        dist_capsule = None
        closest_capsule = closestCapsule((x, y), capsule, walls)
        if closest_capsule is None:
            features["closest-capsule-direction"] = 0
        elif closest_capsule[0] >= x and closest_capsule[1] >= y:
            features["closest-capsule-direction"] = 1
        elif closest_capsule[0] < x and closest_capsule[1] >= y:
            features["closest-capsule-direction"] = 2
        elif closest_capsule[0] < x and closest_capsule[1] < y:
            features["closest-capsule-direction"] = 3
        else:
            features["closest-capsule-direction"] = 4
        if closest_capsule is not None:
            dist_capsule = abs(closest_capsule[0]-x)+abs(closest_capsule[1]-y)


        if dist_food is not None:
            features["closest-food"] = float(dist_food) / (walls.width + walls.height)
        if dist_capsule is not None:
            features["closest-capsule"] = float(dist_capsule) / (walls.width + walls.height)
        return features
        