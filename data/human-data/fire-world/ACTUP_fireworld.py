import os
import pickle
import fnmatch

def load_fireworld_data(human=True,path=''):
    trajectories = []
    if human:
        os.chdir(path)#'/Users/paulsomers/COGLE/gym-gridworld/data/behavior-only'
        for file in os.listdir('.'):
            if fnmatch.fnmatch(file, '*.tj'):
                trajectories.append(pickle.load(open(file, 'rb')))
    return trajectories


data = load_fireworld_data(human=True,path='/Users/paulsomers/COGLE/gym-gridworld/data/human-data/fire-world/')

print("done")