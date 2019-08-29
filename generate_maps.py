from common import generate_gridworld_from_map
from gym_gridworld.envs import create_np_map
import pickle
import imageio
import os
path='./gym_gridworld/'

"""
Generate Maps given initial top left point and width and height

"""


def scale_image(image, scale):
    # Repeats the pattern in order to scale it
    return image.repeat(scale, axis=0).repeat(scale, axis=1)

# x,y  are the coordinates of the 0 point of this map.
def encode_map(x, y, w, h, scale):

    filename = '{}-{}'.format(x, y)

    # Check first if image exists
    # for files in os.listdir(path+'maps'):
    #     if filename+'.mp' == files:
    #         print("File exists!!!")
    #         return

    terrain_dict = generate_gridworld_from_map.load_terrain_file('./data/world_map_terrain.json')
    terrain_slice_dict = {}
    for lon in range(x,x + w): # 70, 60 e.g. 70,50 starting point (0,0) point for the 10x10 map. To real world coords just add 70 to x and 50 to y
        for lat in range(y, y + h): # 50, 60
            terrain_slice_dict[(lon, lat)] = terrain_dict[(lon, lat)]
    print("Reading complete")

    print("saving map.")
    with open(path + 'maps/' + filename + '.mp', 'wb') as handle:
        pickle.dump(terrain_slice_dict, handle)

    map = create_np_map.convert_map_to_volume_dict(x, y, terrain_slice_dict, w, h)

    print("saving map image")
    image = scale_image(map['rgb_image_with_actors'],scale)
    imageio.imwrite(path + 'maps/' + filename + '.png', image)


# There are features missing so you cannot encode the whole map!!! Also the above function wil return an error in this case
#(330,352)
# ms = [ (65,466),(260,241),(291,231),(241,163),(265,311),(224,153),(265,308),(321,337),(167,174),(149,341),(146,456),(385,291),(334,203),(360,112),(20,94),(308,110),(164,90),(171,333)]
# for mp in ms:
#     print(mp)
#     encode_map(mp[0], mp[1],10,10, 5) # 5 recommended scale for 10x10 maps
encode_map(330, 352,10,10, 5)