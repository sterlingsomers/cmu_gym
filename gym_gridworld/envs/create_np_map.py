import os
import pickle
import numpy as np
from gym_gridworld.envs.mapquery import terrain_request
from pathlib import Path
import itertools
path='./gym_gridworld/'


def get_feature_value_maps(x,y,map):
    '''This will create, load, and expand a feature-> value dictionary and
    a value-> feature dictionary.
    Will return 2 dictionaries.'''
    feature_value_map = {}
    value_feature_map = {}
    #first check for existing feature maps
    print("create_np_map.py:get_feature_value_maps cwd {}".format(os.getcwd()))
    feature_to_value = Path(path+'features/features_to_values.dict')
    value_to_feature = Path(path+'features/values_to_features.dict')

    if feature_to_value.is_file():
        feature_value_map = pickle.load(open(feature_to_value,'rb'))
    else:
        raise Exception("Could not find the feature_to_value dictionary")
    if value_to_feature.is_file():
        value_feature_map = pickle.load(open(value_to_feature, 'rb'))
    else:
        raise Exception("Could not find the value_to_feature dictionary")


    return (feature_value_map, value_feature_map)

# BELOW FUMCTION HAS BETTER PERFORMANCE
# def convert_map_to_volume_dict(x,y,map,width,height):
#     return_dict = {}
#     top_left = (x,y)
#     feature_value_map = {}
#     img = np.zeros((width,height,3),dtype=np.uint8)
#     vol = np.zeros((5, width, height))
#     flat = np.zeros((width,height))
#
#     color_map = {'pine tree':[0,100,14],'pine trees':[0,172,23],'grass':[121,151,0],
#                  'bush':[95,98,57],'bushes':[164,203,8],'trail':[145,116,0],
#                  'water':[0,34,255],
#                  'drone':{0:[102,0,51],1:[153,0,153],2:[255,51,255],3:[255,153,255],4:[255,0,0]},
#                  'hiker':[255,0,0]}
#
#     feature_value_map,value_feature_map = get_feature_value_maps(x,y,map)
#     if list(value_feature_map.keys()):
#         value = max(list(value_feature_map.keys())) + 1
#     else:
#         value = 1.0
#
#
#     for xy, feat in map.items():
#         #print(feat[1], feature_value_map.keys())
#         # id = '{} {}'.format(feat[1], feat[0])
#         # if id not in list(feature_value_map.keys()):
#         if feat[1] not in list(feature_value_map.keys()):
#             print('FEATURE NOT IN LIST')
#             #feature_value_map[feat[1]] = {}
#
#             #for i in range(5):
#             feature_value_map[feat[1]] = {'val': value, 'color':color_map[feat[1]]}
#             value_feature_map[value] = {'feature':feat[1], 'alt':float(feat[0]), 'color':color_map[feat[1]]}
#             value += 1
#
#             #value += 20
#         #put it in the flat
#         flat[xy[1] - top_left[1], xy[0] - top_left[0]] = feature_value_map[feat[1]]['val']
#         img[xy[1]- top_left[1], xy[0] - top_left[0], :] = feature_value_map[feat[1]]['color']
#         #project it downwards through the volume
#         for z in range(feat[0],-1,-1):
#             vol[z,xy[1] - top_left[1],xy[0] - top_left[0]] = feature_value_map[feat[1]]['val']
#
#
#
#     return_dict['feature_value_map'] = feature_value_map
#     return_dict['value_feature_map'] = value_feature_map
#     #save before returning
#     #todo fix value_feature_map and feature_maps -> they should be the same (except inside out)
#
#     return_dict['vol'] = vol
#     return_dict['flat'] = flat
#     return_dict['img'] = img
#
#     #add the hiker and the drone
#     feature_value_map['hiker'] = {}
#     feature_value_map['drone'] = {}
#     #drone
#     # value += 20
#     value = max(list(value_feature_map.keys())) + 1
#     for i in range(5):
#         feature_value_map['drone'][i] = {'val': value, 'color': color_map['drone'][i]}
#         value_feature_map[value] = {'feature': 'drone', 'alt':i, 'color':color_map['drone'][i]}
#         value += 1
#
#     #hiker - reserving 50
#     value = 50
#     feature_value_map['hiker']['val'] = value
#     feature_value_map['hiker']['color'] = color_map['hiker']
#     value_feature_map[value] = {'feature':'hiker', 'alt':0, 'color':color_map['hiker']}
#
#
#     return return_dict

def convert_map_to_volume_dict(x,y,map,width,height):

    print("Convert_map_to_volume_dict x {} y {} ".format(x,y))
    return_dict = {}
    top_left = (x,y)
    feature_value_map = {}
    img = np.zeros((width,height,3),dtype=np.uint8)
    vol = np.zeros((5, width, height))
    flat = np.zeros((width,height))

    color_map = {'pine tree':[0,100,14],'pine trees':[0,172,23],'grass':[121,151,0],
                 'bush':[95,98,57],'bushes':[164,203,8],'trail':[145,116,0],
                 'water':[0,34,255],
                 'drone':{0:[102,0,51],1:[153,0,153],2:[255,51,255],3:[255,153,255],4:[255,0,0]},
                 'hiker':[255,0,0]}

    feature_value_map,value_feature_map = get_feature_value_maps(x,y,map)
    if list(value_feature_map.keys()):
        value = max(list(value_feature_map.keys())) + 1
    else:
        value = 1.0

    for xy, feat in map.items():
        # print(feat[1], feature_value_map.keys())
        id = '{} {}'.format(feat[1], feat[0])
        if id not in list(feature_value_map.keys()):
            print('FEATURE NOT IN LIST')
            # feature_value_map[feat[1]] = {}

            # for i in range(5):
            # feature_value_map[id] = {'val': value, 'color':color_map[id]}
            # value_feature_map[value] = {'feature':id[:-2], 'alt':float(id), 'color':color_map[id]}
            # value += 1

            # value += 20
        # put it in the flat
        flat[xy[1] - top_left[1], xy[0] - top_left[0]] = feature_value_map[id]['val']
        img[xy[1] - top_left[1], xy[0] - top_left[0], :] = feature_value_map[id]['color']
        # project it downwards through the volume
        for z in range(feat[0], -1, -1):
            vol[z, xy[1] - top_left[1], xy[0] - top_left[0]] = feature_value_map[id]['val']



    return_dict['feature_value_map'] = feature_value_map
    return_dict['value_feature_map'] = value_feature_map
    #save before returning
    #todo fix value_feature_map and feature_maps -> they should be the same (except inside out)

    return_dict['vol'] = vol
    return_dict['flat'] = flat
    return_dict['img'] = img

    #add the hiker and the drone
    feature_value_map['hiker'] = {}
    feature_value_map['drone'] = {}
    #drone
    # value += 20
    value = max(list(value_feature_map.keys())) + 1
    for i in range(5):
        feature_value_map['drone'][i] = {'val': value, 'color': color_map['drone'][i]}
        value_feature_map[value] = {'feature': 'drone', 'alt':i, 'color':color_map['drone'][i]}
        value += 1

    #hiker - reserving 50
    value = 50#max(list(value_feature_map.keys())) + 20

    #for i in range(5):
    feature_value_map['hiker']['val'] = value
    feature_value_map['hiker']['color'] = color_map['hiker']
    value_feature_map[value] = {'feature':'hiker', 'alt':0, 'color':color_map['hiker']}
    return return_dict

def map_to_volume_dict(path_str, x=0,y=0,width=5,height=5):

    """Looks for a map with filename corresponding to x,y coordinates corresponding to pickled cache.
       If it can't find a version cached on disk it requests a new map.
       Finally it converts it to a volume dictionary and returns it."""

    #does the map already exist in the maps/ folder?

    path = os.path.abspath(path_str)
    print("Looking for maps in {}".format(path))
    return_dict = {}
    filename = '{}-{}.mp'.format(x,y)
    maps = []
    map = 0
    for files in os.listdir(path):
        if files.endswith(".mp"):
            maps.append(files)
    #loops through because I'll need the actual map
    for files in maps:
        if filename == files:
            print("loading existing map {}".format(filename))
            map = pickle.load(open( os.path.join(path,filename),'rb'))

    if not map:
        print("Could not find or was not given a map file.")
        print("Trying to communicate with MAVSIM to pull map remotely.")
        print("(YOU NEED TO SUPPLY A MAP OR HAVE MAVSIM RUNNING!!!)")

        map = terrain_request(x,y,width,height)

        #store it for future use
        print("saving map.")
        with open( os.path.join(path, filename), 'wb') as handle:
            pickle.dump(map, handle)
    #convert_map_to_volume_dict(x,y,map)
    return convert_map_to_volume_dict(x,y,map,width,height)


def create_custom_map(map,offset=(-1,-1)):
    features_to_values, values_to_features = get_feature_value_maps(0, 0, 0)
    if not features_to_values:
        return None
    color_map = {'pine tree': [0, 100, 14], 'pine trees': [0, 172, 23], 'grass': [121, 151, 0],
                 'bush': [95, 98, 57], 'bushes': [164, 203, 8], 'trail': [145, 116, 0],
                 'water': [0, 34, 255],
                 'drone': {0: [102, 0, 51], 1: [153, 0, 153], 2: [255, 51, 255], 3: [255, 153, 255], 4: [255, 0, 0]},
                 'hiker': [255, 0, 0]}
    vol = np.zeros((5,map.shape[0],map.shape[1]))
    # create the img
    img = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
    for layer_num in range(vol.shape[0]):
        if layer_num == 0:
            vol[layer_num] = map
        else:
            #combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
            for x,y in list(itertools.product(range(0,vol.shape[1]), range(0,vol.shape[2]))):
                img[x,y,:] = values_to_features[map[x,y]]['color']
                if layer_num <= values_to_features[map[x,y]]['alt']:
                    vol[layer_num,x,y] = map[x,y]
    layer_num += 1


    #add the drone and hiker to dictionaries
    # add the hiker and the drone
    features_to_values['hiker'] = {}
    features_to_values['drone'] = {}
    # drone
    # value += 20
    value = max(list(values_to_features.keys())) + 1
    for i in range(5):
        features_to_values['drone'][i] = {'val': value, 'color': color_map['drone'][i]}
        values_to_features[value] = {'feature': 'drone', 'alt': i, 'color': color_map['drone'][i]}
        value += 1

    # hiker - reserving 50
    value = 50  # max(list(value_feature_map.keys())) + 20

    # for i in range(5):
    features_to_values['hiker']['val'] = value
    features_to_values['hiker']['color'] = color_map['hiker']
    values_to_features[value] = {'feature': 'hiker', 'alt': 0, 'color': color_map['hiker']}

    return_dict = {}
    return_dict['feature_value_map'] = features_to_values
    return_dict['value_feature_map'] = values_to_features
    return_dict['vol'] = vol
    return_dict['flat'] = map
    return_dict['img'] = img
    return_dict['offset'] = offset

    return return_dict





#sample code
#a = map_to_volume_dict(90,70,10,10)
# f,v = get_feature_value_maps(300,200,a) #300,200
# print('complete.')