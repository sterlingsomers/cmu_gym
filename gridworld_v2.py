'''Returns an image and not a dictionary with ['img'] and ['small']'''
import numpy as np
import random
import itertools
import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt

from gym import spaces


class gameOb(): # Any object registered will have all these properties below!
    def __init__(self, coordinates, size, color, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.color = color
        self.reward = reward
        self.name = name


class gameEnv():
    metadata = {'render.modes': ['human']}
    spec = None
    def __init__(self, partial, size):#, goal_color):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        # MINE
        self.obs_shape = [100, 100, 3]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(self.actions)
        self.reward_range = (-float('inf'), float('inf')) # or self.reward_range = (0,1)
        ####
        self.objects = []
        self.partial = partial
        self.bg = np.zeros([size, size])
        self.num_fires = 3
        self.goal_color = [0,1,0]#[np.random.uniform(), np.random.uniform(), np.random.uniform()]
        self.goal_reward = 1
        self.fire_reward = -1
        self.info = {}
        self.info['fire'] = 0
        self.info['goal'] = 0

    def seed(self, seed=None):
        np.random.seed(seed=seed) # It works the seed now!!!


    def getFeatures(self):
        return np.array([self.objects[0].x, self.objects[0].y]) / float(self.sizeX)

    def reset(self):#, goal_color):
        self.info = {}
        self.info['fire'] = 0
        self.info['goal'] = 0
        self.objects = []
        # self.goal_color = goal_color
        self.other_color = [1 - a for a in self.goal_color]
        self.orientation = 0
        # Below every observation/object is a class gameOb with its properties (coords, size, etc)
        self.hero = gameOb(self.newPosition(0), 1, [0, 0, 1], None, 'hero') # "registering" objects with coords, size, color [R,G,B], reward, name
        self.objects.append(self.hero)
        # for i in range(self.sizeX - 1): # This creates multiple targets (8 if size.X=9) to be collected thats why it doesnt ever terminate.
        bug = gameOb(self.newPosition(0), 1, self.goal_color, self.goal_reward, 'goal')
        self.objects.append(bug)
        for i in range(self.num_fires): # It will create 9-1=8 fires in total
            hole = gameOb(self.newPosition(0), 1, [1,0,0], self.fire_reward, 'fire') # REWARD FOR FIRE!!! CHANGE IT ALSO AT checkgoal function
            self.objects.append(hole)
        # state, s_big = self.renderEnv()
        obs = self.renderEnv()
        s_big = obs['img']
        state = obs['small']
        self.state = state
        return s_big#state, s_big

    def moveChar(self, action):
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - 90 counter-clockwise, 5 - 90 clockwise
        self.info['success'] = True
        hero = self.objects[0]
        blockPositions = [[-1, -1]]
        for ob in self.objects:
            if ob.name == 'block': blockPositions.append([ob.x, ob.y])
        blockPositions = np.array(blockPositions)
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if action < 4:
            if self.orientation == 0:
                direction = action
            if self.orientation == 1:
                if action == 0:
                    direction = 1
                elif action == 1:
                    direction = 0
                elif action == 2:
                    direction = 3
                elif action == 3:
                    direction = 2
            if self.orientation == 2:
                if action == 0:
                    direction = 3
                elif action == 1:
                    direction = 2
                elif action == 2:
                    direction = 0
                elif action == 3:
                    direction = 1
            if self.orientation == 3:
                if action == 0:
                    direction = 2
                elif action == 1:
                    direction = 3
                elif action == 2:
                    direction = 1
                elif action == 3:
                    direction = 0

            if direction == 0 and hero.y >= 1 and [hero.x, hero.y - 1] not in blockPositions.tolist():
                hero.y -= 1
            if direction == 1 and hero.y <= self.sizeY - 2 and [hero.x, hero.y + 1] not in blockPositions.tolist():
                hero.y += 1
            if direction == 2 and hero.x >= 1 and [hero.x - 1, hero.y] not in blockPositions.tolist():
                hero.x -= 1
            if direction == 3 and hero.x <= self.sizeX - 2 and [hero.x + 1, hero.y] not in blockPositions.tolist():
                hero.x += 1
        if hero.x == heroX and hero.y == heroY: # hit a wall and stay at same position
            penalize = 0.0
            self.info['success'] = False
        self.objects[0] = hero
        return penalize, self.info

    def newPosition(self, sparcity):
        iterables = [range(self.sizeX), range(self.sizeY)] # list [range(0,9) range(0,9)]
        points = []
        for t in itertools.product(*iterables):# points will have all the coords pairs [(0,0), (0,1),...,(8,8)]
            points.append(t)
        for objectA in self.objects: # objects is a list containing two gameOb objects with their respective attributes
            if (objectA.x, objectA.y) in points: points.remove((objectA.x, objectA.y)) # Remove from points the coords of the already placed objects (hero and goal)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location] # returns new location

    def checkGoal(self): # This seems that never returns done=True
        hero = self.objects[0]
        others = self.objects[1:]

        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y and hero != other: # if hero overlaps with an object (goal or fire)
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(0), 1, self.goal_color, self.goal_reward, 'goal')) # Moves the goal somewhere else as it is now reached
                    self.info['goal'] = other.reward
                    return other.reward, True, self.info#False
                else: # else its a fire
                    self.objects.append(gameOb(self.newPosition(0), 1, [1,0,0], self.fire_reward, 'fire')) #self.other_color. We keep fire under the red color
                    self.info['fire'] = other.reward
                    return other.reward, True, self.info # return done=true if you want to reset in case you hit a fire
        if ended == False:
            return 0.0, False, self.info
            # return -0.01, False

    def renderEnv(self):
        if self.partial == True:
            padding = 2
            a = np.ones([self.sizeY + (padding * 2), self.sizeX + (padding * 2), 3])
            a[padding:-padding, padding:-padding, :] = 0
            a[padding:-padding, padding:-padding, :] += np.dstack([self.bg, self.bg, self.bg])
        else:
            a = np.zeros([self.sizeY, self.sizeX, 3])
            padding = 0
            a += np.dstack([self.bg, self.bg, self.bg])
        hero = self.objects[0]
        for item in self.objects:
            a[item.y + padding:item.y + item.size + padding, item.x + padding:item.x + item.size + padding,
            :] = item.color
            # if item.name == 'hero':
            #    hero = item
        if self.partial == True:
            a = a[(hero.y):(hero.y + (padding * 2) + hero.size), (hero.x):(hero.x + (padding * 2) + hero.size), :]
        a_big = scipy.misc.imresize(a, [100, 100, 3], interp='nearest') # was 32,32,3
        obs = {}
        obs['img'] = a_big
        obs['small'] = a
        # plt.imshow(a_big)
        # plt.show()
        # # self.fig.show()
        # plt.pause(0.1)
        # plt.close('all')
        return obs# both obs are used in reset!!!#a, a_big # a is probably a non image representation whereas a_big is an image

    def step(self, action):
        # plt.close('all')
        penalty, self.info = self.moveChar(action)
        reward, done, self.info = self.checkGoal()
        info = self.info
        # state, s_big = self.renderEnv()
        obs = self.renderEnv()
        if reward == None:
            print('done=', done)
            print('reward=', reward)
            print('penalty=', penalty)
            # return state, (reward + penalty), done
            return (obs['img'], (reward + penalty), done, info)
        else:
            goal = None
            for ob in self.objects:
                if ob.name == 'goal':
                    goal = ob
            # return state, s_big, (reward + penalty), done, [self.objects[0].y, self.objects[0].x], [goal.y, goal.x]
            return (obs['img'], (reward + penalty), done, info)
