import numpy as np
import pandas as pd
import numpy as np
from pysbrl import BayesianRuleList, train_sbrl
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from mdlp.discretization import MDLP


def compute_intervals(mdlp_discretizer):
    category_names = []
    for i, cut_points in enumerate(mdlp_discretizer.cut_points_):
        idxs = np.arange(len(cut_points) + 1)
        names = mdlp_discretizer.assign_intervals(idxs, i)
        category_names.append(names)
    return category_names

# Load Data
df = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/df_dataframe.df')

# Select and combine columns
data = df[['altitudes','hiker_in_ego','drone_alt']]
data['combined']= data.values.tolist()
v = data['combined'].values
# Format the nested lists into a flat vector
d = np.empty([1978, 7])
for i in range(1978):
    d[i, :] = np.hstack(v[i])

# Correct formats
x = d.astype(int)
# y = df['target'].values
y = df['actions'].values
y = y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
feature_names = ["Far Left object's altitude", "Left object's alt", "Central object's alt",
                  "Right object's alt", "Far Right object's alt", "hiker present", "Drone's alt"]


rule_list = BayesianRuleList(seed=1, feature_names=feature_names, verbose=2)
rule_list.fit(x_train, y_train)
print(rule_list)
print('acc: %.4f' % rule_list.score(x_test, y_test))



# rule_ids, outputs, rule_strings = train_sbrl('./data/ttt_train.out', './data/ttt_train.label',
# max_iters=10000, verbose=1)