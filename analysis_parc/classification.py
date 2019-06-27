import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

import maps
from plotting import get_view_at
from plotting import values_to_features

"""

Drone Trajectory Columns

> print(df.columns.values)

['episode' 'timestep' 'nav_stops' 'agent_type' 'actions' 'action_label'
 'values' 'drone_position' 'drone_alt' 'heading' 'crash' 'fc'
 'altitudes_in_slice' 'hiker_in_ego']

"""

MODE = 'location'
MODE = 'terrain'

SUBSAMPLE = False

MINIMUM_PERCENTAGE_DATA_IN_LEAF = 0.05

df = pd.read_pickle('BoxCanyon_D1910_H1010_100_df.df')

# NOTE We transpose the coordinates here to make it more intuitive to interpret

df['x'] = df['drone_position'].transform( lambda pos: pos[1] )
df['y'] = df['drone_position'].transform( lambda pos: 20-pos[0]-1 )

map_tiles_raw     = maps.name_to_map['box_canyon']

for i in range(9):
    new_col_name = 'v{}'.format(i)
    print("New column: {}".format(new_col_name))
    new_col_values =  [ get_view_at( map_tiles_raw, x, y )[i%3][i//3]
                        for x,y in df['drone_position']
                      ]
    new_col_values = [ values_to_features[val]['feature'] for val in new_col_values]
    df[new_col_name]=new_col_values


# ----------------------------------------------------------------
# Drop OK Final Events with outcome labels
# ----------------------------------------------------------------

drop_ok_episodes_events = \
    df.groupby( 'episode' ).filter( lambda rows: (rows[['crash']].sum() == 0)[0] and (len(rows.index)!=70) ) \
      .reset_index()

drop_ok_episodes_last_event = \
    drop_ok_episodes_events.groupby('episode').tail(1)

drop_ok_episodes_last_event['outcome']='drop_ok'


# ----------------------------------------------------------------
# Crash Final Events with outcome labels
# ----------------------------------------------------------------

crash_episodes_events = \
    df.groupby( 'episode' ).filter( lambda rows: rows[['crash']].sum() > 0 ) \
      .reset_index()

crash_episodes_last_event = \
    crash_episodes_events.groupby('episode').tail(1)

crash_episodes_last_event['outcome']='crashed'



# ----------------------------------------------------------------
# All Final Events with outcome labels
# ----------------------------------------------------------------

all_episodes_last_event = pd.concat( [ drop_ok_episodes_last_event, crash_episodes_last_event ] )


if SUBSAMPLE:
    all_episodes_last_event = all_episodes_last_event.sample(10)



# *****************************************************************
# Location
# *****************************************************************

if MODE=='location':

    # Features

    projected_columns = [ 'x', 'y']

    features =  all_episodes_last_event[ projected_columns ]
    labels  = all_episodes_last_event[ ['outcome'] ]

    # Encoded Features

    label_enc = LabelEncoder().fit(labels)
    labels_enc = label_enc.transform(labels)

    print("Raw labels")
    print(labels)
    print("Encoded labels")
    print(labels_enc)

    dt = DecisionTreeClassifier(min_samples_leaf=0.10) # Require leaves to cover at least 10% of data

    dt.fit(features,labels_enc)

    labels_pred = dt.predict(features)

    acc = accuracy_score( labels_enc, labels_pred)

    print("Accuracy {}".format(acc))


    dot_data = StringIO()

    export_graphviz(dt,
                    out_file=dot_data,
                    feature_names=projected_columns,
                    class_names= label_enc.classes_,
                    filled=True, rounded=True,
                    special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    graph.write_pdf('drone_trajectory_decision_tree.pdf')


# *****************************************************************
# Terrain
# *****************************************************************

if MODE == 'terrain':


    # ----------------------------------------------------------------
    # Features
    # ----------------------------------------------------------------

    projected_columns = [ 'v0', 'v1', 'v2','v3', 'v4', 'v5', 'v6', 'v7', 'v8' ]

    features = all_episodes_last_event[ projected_columns ]
    labels   = all_episodes_last_event[ ['outcome'] ]

    print("\nFeatures before encoding")
    print(features)


    # Encoded Features

    category_values = [1,2,24,25,26]
    category_values_all = [category_values]*len(projected_columns)

    print("\nCategory values for each feature")
    print(category_values_all)
    category_names = [ 'tree', 'grass', 'mountain1', 'mountain2', 'mountain3' ]

    feature_enc = OneHotEncoder() #handle_unknown='ignore', categories=category_values_all)

    feature_enc.fit(features)

    features_enc = feature_enc.transform(features)

    print("\nEncoder features")
    print(feature_enc.categories_)

    print("\nEncoded features")
    print(features_enc.toarray())

    label_enc = LabelEncoder().fit(labels)
    labels_enc = label_enc.transform(labels)

    print("Raw labels")
    print(labels)
    print("Encoded labels")
    print(labels_enc)

    dt = DecisionTreeClassifier(min_samples_leaf=MINIMUM_PERCENTAGE_DATA_IN_LEAF) # Require leaves to cover at least 10% of data

    dt.fit(features_enc,labels_enc)

    labels_pred = dt.predict(features_enc)

    acc = accuracy_score( labels_enc, labels_pred)

    print("Accuracy {}".format(acc))


    dot_data = StringIO()

    export_graphviz(dt,
                    out_file=dot_data,
                    feature_names= feature_enc.get_feature_names(),
                    class_names= label_enc.classes_,
                    filled=True, rounded=True,
                    special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    graph.write_pdf('drone_trajectory_decision_tree.pdf')
