import numpy as np
import pickle
import os
import imageio
import argparse
import glob
import shutil
import re
import matplotlib.pyplot as plt

def pathname_to_offset(pathname):

    """Takes a map filename of the form  xxx_yyy.mp OR xxx-yyy.mp
       and extracts the x and y values and returns them as a pair"""

    basename = os.path.basename(pathname)

    match = re.match( "(\d+)[_-](\d+).mp", basename)
    x = match.group(1)
    y = match.group(2)

    return x,y


from gym_gridworld.envs import create_np_map as CNP

def scale_image(image, scale):
    # Repeats the pattern in order to scale it
    return image.repeat(scale, axis=0).repeat(scale, axis=1)


def convert_directory(source_directory,scale):

    abs_src_dir = os.path.abspath(source_directory)
    print("Current working directory: {}".format(os.getcwd()))

    glob_spec = os.path.join( abs_src_dir,'*.mp')
    print("Source path",glob_spec)
    pathnames = glob.glob(glob_spec)


    abs_dest_dir = os.path.join(abs_src_dir,'converted')
    print("Destination path",abs_dest_dir)

    if os.path.exists(abs_dest_dir):
        shutil.rmtree(abs_dest_dir)
    os.mkdir(abs_dest_dir)

    n = len(pathnames)

    for i, pathname in enumerate(pathnames):

        basename = os.path.basename(pathname)

        x,y = filename_to_offset(basename)

        print("file# {} of {} name: {} x:{} y:{}".format(i, n, basename,x,y))

        map_array = pickle.load(open(pathname, 'rb'))

        map = CNP.create_custom_map(map_array,offset=(x,y)) # This creates an 'img' structure in the map dictionary

        outfile_name = '{}-{}'.format(x,y)
        print("saving map to ", os.path.join( abs_dest_dir, outfile_name + '.mp'))

        pickle_out = open( os.path.join( abs_dest_dir, outfile_name + '.mp'), 'wb')
        pickle.dump(map, pickle_out)
        pickle_out.close()

        print("saving map image")
        image = scale_image(map['img'],scale)
        #plt.imshow(image)
        #plt.show()
        imageio.imwrite( os.path.join(abs_dest_dir, outfile_name + '.png'), image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert numpy arrays pickle files to CMU maps')
    parser.add_argument('dir', help='directory of python pickle files to convert')

    args = parser.parse_args()

    print("Parsed args dir={}".format(args.dir))

    scale = 5 # Because we have 5 pixel wide tiles in our maps

    convert_directory( args.dir, scale )
