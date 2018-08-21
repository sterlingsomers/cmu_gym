import pickle

feats = pickle.load(open("./gym_gridworld/envs/features/features_to_values.dict", "rb"))
vals = pickle.load(open("./gym_gridworld/envs/features/values_to_features.dict", "rb"))

print("ok")