import pandas as pd
import numpy as np
from pysbrl import BayesianRuleList, train_sbrl
#from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from mdlp.discretization import MDLP
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def compute_intervals(mdlp_discretizer):
    category_names = []
    for i, cut_points in enumerate(mdlp_discretizer.cut_points_):
        idxs = np.arange(len(cut_points) + 1)
        names = mdlp_discretizer.assign_intervals(idxs, i)
        category_names.append(names)
    return category_names

# Load Data
df = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/df_dataframe.df')
# Check for imbalanced data set
df['actions'].value_counts()

def predict_action(df):
    # Select and combine columns
    data = df[['altitudes','hiker_in_ego','drone_alt']] # altitude vectors contain 5 features.  Drone_alt might need one hot encoding
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
    feature_names = ["Far Left object's altitude", "Left object's altitude", "Central object's altitude",
                      "Right object's altitude", "Far Right object's altitude", "hiker present", "Drone's altitude"]

    class_names = ['down-left','down-diag-left','down-forward','down-diag-right','down-right',
                   'left', 'diag-left', 'forward', 'diag-right', 'right',
                   'up-left', 'up-diag-left', 'up-forward', 'up-diag-right', 'up-right',
                   'drop']
    rule_list = BayesianRuleList(seed=1, feature_names=feature_names, verbose=2)
    rule_list.fit(x_train, y_train)

    class_rf = RandomForestClassifier(n_estimators=100, max_depth=30,
                                    random_state=2)
    class_rf.fit(x_train, y_train)
    y_pred_rf = class_rf.predict(x_test)
    y_pred_rl = rule_list.predict(x_test)
    print(rule_list)
    print('BRL Accuracy: %.4f' % rule_list.score(x_test, y_test))
    print('Random Forest Accuracy: %.4f' % class_rf.score(x_test, y_test))

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred_rl, classes=np.array(class_names), normalize=True,
                          title='Normalized BRL confusion matrix')
    plot_confusion_matrix(y_test, y_pred_rf, classes=np.array(class_names), normalize=True,
                          title='Normalized RF confusion matrix')

    plt.show()

def predict_value(df):
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
    bins = 7 # number of bins
    labels = [i for i in range(bins)]
    y = pd.cut(df['values'], bins, labels=labels) # Create the binning for the continous target vartiable
    y = y.values
    y = np.array(y).astype(int)

    # For debugging, plot the histogram
    # plt.hist(y, bins=4)
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    feature_names = ["Far Left object's altitude", "Left object's altitude", "Central object's altitude",
                      "Right object's altitude", "Far Right object's altitude", "hiker present", "Drone's altitude"]


    rule_list = BayesianRuleList(seed=1, feature_names=feature_names, verbose=2)
    rule_list.fit(x_train, y_train)

    regr_rf = RandomForestRegressor(n_estimators=100, max_depth=30,
                                    random_state=2)
    regr_rf.fit(x_train, y_train)

    class_rf = RandomForestClassifier(n_estimators=100, max_depth=30,
                                    random_state=2)
    class_rf.fit(x_train, y_train)

    print(rule_list)
    print('BRL Accuracy: %.4f' % rule_list.score(x_test, y_test))
    print('Random Forest Regr-Accuracy: %.4f' % regr_rf.score(x_test, y_test))
    print('Random Forest Class-Accuracy: %.4f' % class_rf.score(x_test, y_test))


predict_action(df)
# predict_value(df)
# rule_ids, outputs, rule_strings = train_sbrl('./data/ttt_train.out', './data/ttt_train.label',
# max_iters=10000, verbose=1)