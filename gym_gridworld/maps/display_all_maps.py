import matplotlib.pyplot as plt
import matplotlib.image as mpi
import os,glob

cwd = os.getcwd()

ncol = 5
nrow = 5

selectable_maps = [ (265, 308), (20, 94), (146, 456), (149, 341), (164, 90), (167, 174),
                    (224,153), (241,163), (260,241), (265,311), (291,231),
                    (308,110), (334,203), (360,112), (385,291), (330,352), (321,337)    ]

print("Starting in directory ",cwd)
fig, axs = plt.subplots(nrow,ncol)

fig.subplots_adjust(hspace = 0.3, wspace=.3)

fig.suptitle('All maps and those used for training (train)')

for i, img_file in enumerate(glob.glob("*.png") ):

    stem = img_file.split(".")[0]

    coordinates = tuple( [int(s) for s in stem.split("-") if s.isdigit] )

    print("Image {} named {} with tuple {}".format(i,img_file,tuple))

    img=mpi.imread(img_file)


    ax=axs[i//ncol][i%ncol]
    ax.imshow(img)

    title = stem
    if coordinates in selectable_maps:
        title=title+" (train)"

    ax.set_title(title)
    ax.axis('off')
    ax.title.set_fontsize(8)

plt.show()

