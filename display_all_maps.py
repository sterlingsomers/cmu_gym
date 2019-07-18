import matplotlib.pyplot as plt
import matplotlib.image as mpi
import os,glob
import argparse
import math
import random


MAX_NUM_TO_DISPLAY=49

def plot_all_maps(dir):

    print("Parsed args dir={}".format(dir))
    cwd = os.getcwd()

    print("Current working directory {}".format(cwd))


    #selectable_maps = [ (265, 308), (20, 94), (146, 456), (149, 341), (164, 90), (167, 174),
    #                    (224,153), (241,163), (260,241), (265,311), (291,231),
    #                    (308,110), (334,203), (360,112), (385,291), (330,352), (321,337)    ]

    filter = dir+"/*.png"

    print("Glob filter ",filter)

    files = glob.glob(filter)
    num_files = len(files)
    print("Found {} files".format(num_files))


    if num_files > MAX_NUM_TO_DISPLAY:
        title = "Subsample of {} maps out of {} from {}".format(MAX_NUM_TO_DISPLAY, num_files, dir)
        num_files=MAX_NUM_TO_DISPLAY
        files = random.sample(files,num_files)
    else:
        title = "All {} maps from {}".format(num_files,dir)

    ncol = 7
    nrow = int(math.ceil(num_files/ncol))

    fig, axs = plt.subplots(nrow,ncol)

    fig.subplots_adjust(hspace = 0.3, wspace=.3)


    fig.suptitle(title)



    for i, img_file in enumerate( files):

        if i<num_files:

            print("Loading #{} {}".format(i,img_file))
            img=mpi.imread(img_file)

            img_file=os.path.basename(img_file)
            stem = img_file.split(".")[0]
            coordinates = tuple( [int(s) for s in stem.split("-") if s.isdigit] )

            print("Image {} named {} with tuple {}".format(i,img_file,tuple))



            ax=axs[i//ncol][i%ncol]
            ax.imshow(img)

            title = stem
            ax.set_title(title)


    for i in range(nrow):
        for j in range(ncol):

            ax=axs[i][j]

            ax.axis('off')
            ax.title.set_fontsize(8)


    plt.show()



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Convert numpy arrays pickle files to CMU maps')
    parser.add_argument('dir', help='directory of python pickle files to convert')

    args = parser.parse_args()

    plot_all_maps(args.dir)