
                                        #  R    G    B
WATER_FEATURE = {'val': 15.0, 'color': [   0,  34, 255 ] }   # Bright Blue

TREES_FEATURE = {'val':  3.0, 'color': [   0, 172,  23 ] }   # Bright Kelly Green
GRASS_FEATURE = {'val':  2.0, 'color': [ 121, 151,   0 ] }   # Greeny Yellow

ROAD_FEATURE  = {'val':  5.0, 'color': [ 145, 116,   0 ] }

FIRE_WATCH_TOWER = \
                {'val': 17.0, 'color': [ 160, 160, 160 ] }
FLIGHT_TOWER =  {'val': 13.0, 'color': [ 160, 160, 160 ] }   # Light Grey -- wsame as mountain ridge 2
RUNWAY =        {'val': 12.0, 'color': [  95,  98,  57 ] }

feature_value_map = {

     'water 0': WATER_FEATURE,
     'River':   WATER_FEATURE,
     'Ocean':   WATER_FEATURE,

     'grass 0':      GRASS_FEATURE,
     'Terrain'     : GRASS_FEATURE,

     'pine tree 1':  TREES_FEATURE,
     'pine trees 1': TREES_FEATURE,
     'Oak Trees'   : TREES_FEATURE,
     'bush 0':       {'val': 4.0, 'color': [ 95,  98, 57] },

     'mountain ridge 0': {'val': 22.0, 'color': [250, 250, 250 ] }, # Off White
     'mountain ridge 1': {'val': 24.0, 'color': [200, 200, 200 ] }, # Dark White
     'mountain ridge 2': {'val': 25.0, 'color': [150, 150, 150 ] }, # Light Grey
     'mountain ridge 3': {'val': 26.0, 'color': [100, 100, 100 ] }, # Dark Grey
     'mountain ridge 4': {'val': 31.0, 'color': [ 50,  50,  50 ] }, # Black

     'trail 0': ROAD_FEATURE,
     'Road':    ROAD_FEATURE,

     'shore bank 0': {'val': 6.0, 'color': [95, 98, 57]},
     'bushes 0': {'val': 7.0, 'color': [95, 98, 57]},
     'white Jeep 0': {'val': 8.0, 'color': [95, 98, 57]},
     'unstripped road 0': {'val': 9.0, 'color': [95, 98, 57]},
     'stripped road 0': {'val': 10.0, 'color': [95, 98, 57]},
     'blue Jeep 0': {'val': 11.0, 'color': [95, 98, 57]},
     'runway 0': RUNWAY,
     'runway' : RUNWAY,
     'Flight Tower': FLIGHT_TOWER,
     'flight tower 2': FLIGHT_TOWER,
     'flight tower 0': {'val': 14.0, 'color': [95, 98, 57]},
     'family tent 0': {'val': 16.0, 'color': [95, 98, 57]},
     'firewatch tower 2': FIRE_WATCH_TOWER,
     'Firewatch Tower' : FIRE_WATCH_TOWER,
     'firewatch tower 0': {'val': 18.0, 'color': [95, 98, 57]},
     'large hill 1': {'val': 19.0, 'color': [0, 100, 14]},
     'large hill 0': {'val': 20.0, 'color': [95, 98, 57]},
     'solo tent 0': {'val': 21.0, 'color': [95, 98, 57]},
     'inactive campfire ring 0': {'val': 23.0, 'color': [95, 98, 57]},
     'box canyon 0': {'val': 27.0, 'color': [95, 98, 57]},
     'box canyon 2': {'val': 28.0, 'color': [160, 160, 160]},
     'box canyon 3': {'val': 29.0, 'color': [0, 0, 0]},
     'box canyon 1': {'val': 30.0, 'color': [0, 100, 14]},
     'small hill 1': {'val': 32.0, 'color': [0, 100, 14]},
     'active campfire ring 0': {'val': 33.0, 'color': [255, 161, 0]},
     'cabin 0': {'val': 34.0, 'color': [95, 98, 57]}
}

value_feature_map = {

    1.0:  {'feature': 'pine tree', 'alt': 1.0, 'color': [0, 100, 14]},
    3.0:  {'feature': 'pine trees', 'alt': 1.0, 'color': [0, 172, 23]},

    4.0:  {'feature': 'bush', 'alt': 0.0, 'color': [95, 98, 57]},
    7.0:  {'feature': 'bushes', 'alt': 0, 'color': [95, 98, 57]},
    2.0:  {'feature': 'grass', 'alt': 0.0, 'color': [121, 151, 0]},

    5.0:  {'feature': 'trail', 'alt': 0.0, 'color': [145, 116, 0]},
    6.0:  {'feature': 'shore bank', 'alt': 0, 'color': [95, 98, 57]},
    8.0:  {'feature': 'white Jeep', 'alt': 0, 'color': [95, 98, 57]},

    9.0:  {'feature': 'unstripped road', 'alt': 0, 'color': [95, 98, 57]},
    10.0: {'feature': 'stripped road', 'alt': 0, 'color': [95, 98, 57]},
    11.0: {'feature': 'blue Jeep', 'alt': 0, 'color': [95, 98, 57]},

    12.0: {'feature': 'runway', 'alt': 0, 'color': [95, 98, 57]},
    13.0: {'feature': 'flight tower', 'alt': 2, 'color': [160, 160, 160]},
    14.0: {'feature': 'flight tower', 'alt': 0, 'color': [95, 98, 57]},

    15.0: {'feature': 'water', 'alt': 0, 'color': [0, 34, 255]},

    16.0: {'feature': 'family tent', 'alt': 0, 'color': [95, 98, 57]},
    17.0: {'feature': 'firewatch tower', 'alt': 2, 'color': [160, 160, 160]},
    18.0: {'feature': 'firewatch tower', 'alt': 0, 'color': [95, 98, 57]},
    19.0: {'feature': 'large hill', 'alt': 1, 'color': [0, 100, 14]},
    20.0: {'feature': 'large hill', 'alt': 0, 'color': [95, 98, 57]},
    21.0: {'feature': 'solo tent', 'alt': 0, 'color': [95, 98, 57]},
    23.0: {'feature': 'inactive campfire ring', 'alt': 0, 'color': [95, 98, 57]},

    22.0: {'feature': 'mountain ridge', 'alt': 0, 'color': [ 200, 200, 200 ]},  # Off white
    24.0: {'feature': 'mountain ridge', 'alt': 1, 'color': [ 150, 150, 150 ]},  # Light Gray
    25.0: {'feature': 'mountain ridge', 'alt': 2, 'color': [ 100, 100, 100 ]},  # Gray
    26.0: {'feature': 'mountain ridge', 'alt': 3, 'color': [  50,  50,  50 ]},  # Dark Gray
    31.0: {'feature': 'mountain ridge', 'alt': 4, 'color': [   0,   0,   0 ]},  # Black

    27.0: {'feature': 'box canyon', 'alt': 0, 'color': [95, 98, 57]},
    28.0: {'feature': 'box canyon', 'alt': 2, 'color': [160, 160, 160]},
    29.0: {'feature': 'box canyon', 'alt': 3, 'color': [0, 0, 0]},
    30.0: {'feature': 'box canyon', 'alt': 1, 'color': [0, 100, 14]},

    32.0: {'feature': 'small hill', 'alt': 1, 'color': [0, 100, 14]},
    33.0: {'feature': 'active campfire ring', 'alt': 0, 'color': [255, 161, 0]},
    34.0: {'feature': 'cabin', 'alt': 0, 'color': [95, 98, 57]}
}
