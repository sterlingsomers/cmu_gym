"""Looks at the all future states distributions conditioned on action at a particular state.
   This allows you to ask how much an action choice at the current state changes the state distribution going forward.
   In some cases it may make a small local difference, but in other cases it might totally change the trajectory.
   ToDo: Should really optimize this by managing indexes and using joins."""

import pandas as pd
import matplotlib.pyplot as plt
import queries

import numpy as np

SUPPRESS_COUNTS_LESS_THAN = 10

#x_query =10; y_query = 9
#x_query =6; y_query = 9
#x_query= 1; y_query=0
x_query =9; y_query = 9
x_query=14; y_query = 9

df = pd.read_pickle('BoxCanyon_D1910_H1010_100_df.df')

print("Columns")
print( df.columns.values )


print("Action Names")


action_names = df[['actions','action_label','timestep']].groupby(['actions','action_label']).count() \
    .rename(columns = {'timestep':'count'}).reset_index().sort_values('count',ascending=False)

print("-------------------------------------------------------------------------")
print("Action Names")
print("-------------------------------------------------------------------------")
print(action_names)

def create_test_df():

    rows = [
        # EPS   #POS #HDG #ACT
        [   0, (0,0),   0,   1],
        [   0, (1,0),   0,   2],
        [   0, (2,0),   0,   3],
        [   0, (3,0),   0,   4],

        [   1, (0,0),   0,   8],
        [   1, (1,0),   0,   9],
        [   1, (2,0),   0,   10],
        [   1, (3,0),   0,   11],
        [   1, (3,0),   0,   12],

    ]
    df = pd.DataFrame( rows, columns = ['episode','drone_position','heading','actions'])

    return df

# df = create_test_df()




projdf = df[['episode', 'drone_position', 'heading','actions']]
projdf['x'] = projdf['drone_position'].apply(lambda row: row[0])
projdf['y'] = projdf['drone_position'].apply(lambda row: row[1])
projdf['index']=projdf.index


print("-------------------------------------------------------------------------")
print("Dataframe with x and y columns")
print("-------------------------------------------------------------------------")
print(projdf.head(20))




query_state_idxs = (projdf['x'] == x_query) & (projdf['y'] == y_query)


print("-------------------------------------------------------------------------")
print("Episodes containing state x:{} y:{}".format(x_query,y_query))
print("-------------------------------------------------------------------------")

print(projdf.loc[query_state_idxs].head(20))

first_occurence_of_query = projdf[query_state_idxs] \
                             [['episode', 'index', 'actions']] \
                             .groupby(['episode']).min() \
                             .rename(columns={'actions':'first_action'}) \
                             .reset_index()


print("-------------------------------------------------------------------------")
print("First occurence of state {} {} by episode ".format(x_query,y_query))
print("-------------------------------------------------------------------------")

print( first_occurence_of_query.head(20) )

def is_successor(row):
    eps = row['episode']
    first_values = first_occurence_of_query['index'][first_occurence_of_query['episode']==eps].values
    if len(first_values)>0:
        idx = row['index']
        first = first_values[0]
        return idx > first
    else:
        return False



def get_first_action(row):
    eps = row['episode']
    first_action = first_occurence_of_query['first_action'][first_occurence_of_query['episode']==eps].values
    if len(first_action)>0:
        return first_action[0]
    else:
        return -1

projdf['successor'] = projdf.apply( is_successor, axis=1 )

projdf['first_action'] = projdf.apply( get_first_action, axis=1 )



successors = projdf[ projdf['successor']==True ] [['first_action','episode','actions','x','y','index']]


print("-------------------------------------------------------------------------")
print("Successors of state {} {} by episode ".format(x_query,y_query))
print("-------------------------------------------------------------------------")

print( successors.head(40) )


hist = successors[['first_action','x','y','index']] \
                 .groupby(['first_action','x','y']).count()    \
                 .rename(columns = {'index':'count'}) \
                 .reset_index()



hist = hist[hist['count'] >= SUPPRESS_COUNTS_LESS_THAN]


print("-------------------------------------------------------------------------")
print("Histogram of Successors of state {} {} by first_action, suppress counts < {} ".format(x_query,y_query,SUPPRESS_COUNTS_LESS_THAN))
print("-------------------------------------------------------------------------")

print( hist )



print("-------------------------------------------------------------------------")
print("Totals of Successors of state {} {} by first_action , suppress counts < {}".format(x_query,y_query,SUPPRESS_COUNTS_LESS_THAN))
print("-------------------------------------------------------------------------")


action_subtotals = hist.groupby(["first_action"])
action_subtotals = action_subtotals.sum().reset_index()[['first_action','count']]

print(action_subtotals)
#
# .apply(lambda x: 100*x/float(x['sum']))

def normalize(row):
    actionid = row['first_action']
    sum = action_subtotals[ action_subtotals['first_action']==actionid ]['count'].values[0]
    count = row['count']
    return count/sum

hist['percentage'] = hist.apply( normalize, axis = 1)


print("-------------------------------------------------------------------------")
print("Distribution of Successors of state {} {} by action , suppress counts < {}".format(x_query,y_query,SUPPRESS_COUNTS_LESS_THAN))
print("-------------------------------------------------------------------------")

print(hist.round(3).to_string())

def temp(vals):

    s = np.std(vals)


    return s

var_percentage = hist.groupby(['x','y'], as_index=False).agg( {'percentage':temp} )\
           .rename(columns={'percentage':'var_in_percentage'})


var_count = hist.groupby(['x','y'], as_index=False).agg( {'count':temp} ) \
    .rename(columns={'count':'var_in_count'})

print("-------------------------------------------------------------------------")
print("Variance across actions for state {} {} by action , suppress counts < {}".format(x_query,y_query,SUPPRESS_COUNTS_LESS_THAN))
print("-------------------------------------------------------------------------")

print('by percentage:',var_percentage)
print('by count:',var_count)

print("-------------------------------------------------------------------------")
print("Mean Variance across actions for state {} {} by action , suppress counts < {}".format(x_query,y_query,SUPPRESS_COUNTS_LESS_THAN))
print("-------------------------------------------------------------------------")

print('by percentage:',var_percentage.agg( {'var_in_percentage':np.mean}))
print('by count:',var_count.agg( {'var_in_count':np.mean}))