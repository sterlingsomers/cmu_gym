"""Utilities for sampling, transforming and plotting CMU Drone trajectory datasets

Title type definitions
   Tile: 1.0 Name: pine tree Color: [0, 100, 14]  Altitude: 1.0
   Tile: 2.0 Name: grass Color: [121, 151, 0]  Altitude: 0.0
   Tile: 3.0 Name: pine trees Color: [0, 172, 23]  Altitude: 1.0
   Tile: 4.0 Name: bush Color: [95, 98, 57]  Altitude: 0.0
   Tile: 5.0 Name: trail Color: [145, 116, 0]  Altitude: 0.0
   Tile: 6.0 Name: shore bank Color: [95, 98, 57]  Altitude: 0
   Tile: 7.0 Name: bushes Color: [95, 98, 57]  Altitude: 0
   Tile: 8.0 Name: white Jeep Color: [95, 98, 57]  Altitude: 0
   Tile: 9.0 Name: unstripped road Color: [95, 98, 57]  Altitude: 0
   Tile: 10.0 Name: stripped road Color: [95, 98, 57]  Altitude: 0
   Tile: 11.0 Name: blue Jeep Color: [95, 98, 57]  Altitude: 0
   Tile: 12.0 Name: runway Color: [95, 98, 57]  Altitude: 0
   Tile: 13.0 Name: flight tower Color: [160, 160, 160]  Altitude: 2
   Tile: 14.0 Name: flight tower Color: [95, 98, 57]  Altitude: 0
   Tile: 15.0 Name: water Color: [0, 34, 255]  Altitude: 0
   Tile: 16.0 Name: family tent Color: [95, 98, 57]  Altitude: 0
   Tile: 17.0 Name: firewatch tower Color: [160, 160, 160]  Altitude: 2
   Tile: 18.0 Name: firewatch tower Color: [95, 98, 57]  Altitude: 0
   Tile: 19.0 Name: large hill Color: [0, 100, 14]  Altitude: 1
   Tile: 20.0 Name: large hill Color: [95, 98, 57]  Altitude: 0
   Tile: 21.0 Name: solo tent Color: [95, 98, 57]  Altitude: 0
   Tile: 22.0 Name: mountain ridge Color: [95, 98, 57]  Altitude: 0
   Tile: 23.0 Name: inactive campfire ring Color: [95, 98, 57]  Altitude: 0
   Tile: 24.0 Name: mountain ridge Color: [0, 100, 14]  Altitude: 1
   Tile: 25.0 Name: mountain ridge Color: [160, 160, 160]  Altitude: 2
   Tile: 26.0 Name: mountain ridge Color: [0, 0, 0]  Altitude: 3
   Tile: 27.0 Name: box canyon Color: [95, 98, 57]  Altitude: 0
   Tile: 28.0 Name: box canyon Color: [160, 160, 160]  Altitude: 2
   Tile: 29.0 Name: box canyon Color: [0, 0, 0]  Altitude: 3
   Tile: 30.0 Name: box canyon Color: [0, 100, 14]  Altitude: 1
   Tile: 31.0 Name: mountain ridge Color: [0, 0, 0]  Altitude: 4
   Tile: 32.0 Name: small hill Color: [0, 100, 14]  Altitude: 1
   Tile: 33.0 Name: active campfire ring Color: [255, 161, 0]  Altitude: 0
   Tile: 34.0 Name: cabin Color: [95, 98, 57]  Altitude: 0
   """

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import maps


# Load Feature Dictionaries and Alter Them a Little (to fix bugs and to improve visibility)

features_to_values = pd.read_pickle('features_to_values.dict')
values_to_features = pd.read_pickle('values_to_features.dict')

#print("Title type definitions")
for tileid,props in values_to_features.items():
    #print("   Tile: {} Name: {} Color: {}  Altitude: {}".format(tileid, props['feature'],props['color'],props['alt']))
    if tileid ==24:
        values_to_features[tileid]['color']=[200,200,200]
    if tileid ==26:
        values_to_features[tileid]['color']=[80,80,80]


def tiles_to_altitudes(tiles):

    """Converts a world described by tileids to a map of altitudes.
       This works because every tile type has exactly one altitude.

       :param tiles: 2D numpy array of tileids
       :return 2D numpy array of altitudes
    """

    altitudes = np.zeros(tiles.shape)

    for i in range( altitudes.shape[0]):
        for j in range( altitudes.shape[1]):
            tile_code = tiles[i][j]
            #if tile_code ==27:
            #    tile_code=26 # Convert to mountain ridge for now

            altitudes[i][j]=values_to_features[tile_code]['alt']

    return altitudes


def tiles_to_rgb(tiles):

    """Converts a world described by tileids into an RGB image

        :param tiles: 2D numpy array of tileids
        :return 3D numpy array of the form row,col,channel giving RGB color values at each cell"""

    img = np.zeros( [tiles.shape[0],tiles.shape[1],3])
    for i in range( tiles.shape[0] ):
        for j in range( tiles.shape[1] ):
            C = values_to_features[ tiles[i][j]]['color']
            img[i,j,0] = C[0]/256.0
            img[i,j,1] = C[1]/256.0
            img[i,j,2] = C[2]/256.0

    return img


def has_crashed(crash_flags):

    """returns true if the drone crashed during the run

        :param crash_flags  a numpy vector of crash_flags which is assumed to be all zero unless there was a crash
        :return True if the drone crashed"""

    return np.any( np.array(crash_flags)!=0 )


def is_stuck(crash_flags):

    """returns true if the drone got stuck looping and failed to terminate in 70 steps.
       ToDo: Assumes that the number 70 is the cutoff point!! This is a brittle condition
       :param crash_flags
       :return True if drone failed to terminate before 70 steps
       """
    return len(crash_flags)==70



def get_view_at( map_tiles, x_native, y_native ):

    """Returns the 3x3 view around the agent at position x_native, y_native.
       """

    DEFAULT_TILE_ID =33

    xmin = max(x_native-1,0); xmax = min(map_tiles.shape[1]-1,x_native+1)
    ymin = max(y_native-1,0); ymax = min(map_tiles.shape[0]-1,y_native+1)

    view = np.ones( (3,3), dtype=np.int32 ) * DEFAULT_TILE_ID
    for x in np.arange( xmin, xmax+1 ):
        for y in np.arange( ymin, ymax+1 ):
            view[x-(x_native-1)][y-(y_native-1)] = map_tiles[ x, y ]

    return view



def plot_trajectory(trajectory_data_frame_df, episode, title="Trajectory", map_name="box_canyon"):

    """Given a pandas dataframe of events from a training run covering multiple episodes,
       and a desired episode number,
       plots the trajectory of the episode in various ways to provide an understanding of the drone behavior.

       :param trajectory_data_frame_df - pandas dataframe of simulation events
       :param episode - an integer starting from zero indicating which episode to plot
       :param title - title for top of plots
       :param map_name - map to use as background for the visualizations and feature generation

       ToDo: This routine should be broken up to make the individual graphs usable.
             We should probably standardize the interface on something close to the raw data so
             that routines can be easily applied.
       """

    # Extract single episode from all data

    episode_events_df = trajectory_data_frame_df[  trajectory_data_frame_df['episode'] == episode  ]

    x_end_raw, y_end_raw  = list(episode_events_df['drone_position'])[-1]
    x_start_raw, y_start_raw  = list(episode_events_df['drone_position'])[0]


    # Background map data

    map_tiles_raw     = maps.name_to_map[map_name]
    map_altitudes_raw = tiles_to_altitudes(map_tiles_raw)


    # Flip vertically so that imshow will render this the way we expect on the screen

    map_tiles_flip     = np.flipud(map_tiles_raw)  # np.transpose( np.flipud(map_tiles_raw)     )
    map_altitudes_flip = np.flipud(map_altitudes_raw)


    # Flip and transpose so that 3D Bar chart shows expected landscape

    map_tiles_ftp     =  np.transpose( np.flipud(map_tiles_raw)     )
    map_altitudes_ftp = np.transpose( np.flipud(map_altitudes_raw) )


    # Extract Drone Coordinates from XYd in a format plotting routines like
    # !!! We are flipping and transposing coordinates here

    XYd = episode_events_df['drone_position']


    # Need to swap x and y, because original values are row,col and we need x,y
    # Also, we need to flip y vertically because the origin 0 is at the bottom in pyplot

    Xd  = np.array( [ y for x, y in XYd ] )
    Yd  = np.array( [ map_tiles_raw.shape[1] - x - 1 for x, y in XYd ] )

    Zd  = episode_events_df['drone_alt']

    T   = range(len(XYd))   # Time stamps

    crash_flags = episode_events_df['crash']

    crashed = has_crashed(crash_flags)

    stuck = is_stuck(crash_flags)


    #-------------------------------------------------
    # Setup Plotting
    #-------------------------------------------------

    fig = plt.figure( figsize=(16,8) )

    prows = 2 # Number of rows in plotted graphs
    pcols = 4 # Number of columns in plotted graphs

    pnum=1


    #-------------------------------------------------
    # 2D Tile Type Map
    #-------------------------------------------------

    ax = fig.add_subplot(prows, pcols, pnum); pnum=pnum+1

    img = tiles_to_rgb(map_tiles_flip)

    plt.imshow(img)

    plt.plot(Xd,Yd,'r-')
    plt.plot(Xd,Yd,'r.')


    plt.plot([Xd[0]],[Yd[0]],'ko' )
    plt.plot([Xd[-1]],[Yd[-1]],'kx')


    plt.xlim( [0,20])
    plt.ylim( [0,20])
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('XY Trajectory Start: {},{} End: {},{}'.format(
                     y_start_raw, map_tiles_raw.shape[0]-x_start_raw-1,
                     y_end_raw, map_tiles_raw.shape[0]-x_end_raw-1     ))


    #-------------------------------------------------
    # Space-Time Plot to Visualize Motion Dynamics
    #-------------------------------------------------

    ax = fig.add_subplot(prows, pcols, pnum,projection='3d'); pnum=pnum+1

    ax.plot(Xd,Yd,T,'r-')
    ax.plot(Xd,Yd,T,'r.')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Time', rotation=90)

    ax.set_xlim([0, map_tiles_raw.shape[0]])
    ax.set_ylim([0, map_tiles_raw.shape[1]])

    plt.plot([Xd[0]],[Yd[0]],[T[0]],'ko' )
    plt.plot([Xd[-1]],[Yd[-1]],[T[-1]],'kx')

    plt.title("Location-Timestep plot")


    #-------------------------------------------------
    # 3D Terrain Plot Using Bars
    #-------------------------------------------------


    # First Create Bars for terrain over domain Xt,Yt with height Zt and base height Bt

    ax = fig.add_subplot(prows, pcols, pnum,projection='3d'); pnum=pnum+1

    Xt, Yt = np.meshgrid(
        range( map_altitudes_ftp.shape[0] ),
        range( map_altitudes_ftp.shape[1] )  )  # mesh-grid creates a 2D array of X and Y values

    Xt = Xt.ravel()
    Yt = Yt.ravel()

    Zt = np.zeros_like(Xt)
    Ct = list()

    for i in range(Xt.shape[0]):   # Walk over each cell and set Zd and color

        Zt[i] = map_altitudes_ftp[Xt[i]][Yt[i]]
        col   = values_to_features[  map_tiles_ftp[Xt[i]][Yt[i]] ]['color'];
        Ct.append( [ float(i)/256.0 for i in col ] )

    Bt = np.zeros_like(Zt)   # Base of terrain is Zd 0 for all cells


    # Drone Trajectory Points

    Ad = np.ones_like(Xd)       # Altitude of the block above baseline, but base line is height of block above floor

    DRONE_COLOR = [ 1, 0.4, 0.4, 1 ]
    Cd = list()
    for i in range( Xd.shape[0]):
        Cd.append( DRONE_COLOR )


    # Combine Terrain and Drone Points

    Xc = np.concatenate( [ Xt, Xd   ] )
    Yc = np.concatenate( [ Yt, Yd   ] )
    Zc = np.concatenate( [ Zt, Ad   ] )    # Altitude is just a unit above base
    Bc = np.concatenate( [ Bt, Zd-1 ] )    # Base to draw bar at altitude above floor

    Cc = list();  Cc.extend(Ct);  Cc.extend(Cd)


    # Plot Points and Annotations

    BAR_WIDTH_PX = 1

    ax.bar3d(Xc, Yc, Bc, BAR_WIDTH_PX, BAR_WIDTH_PX, Zc, color=Cc, shade=True)

    ax.plot( [0],[0],[20],'w')   # Force height to be the same as width and depth to create regular cube pixels

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Alt')

    ax.set_xlim([0,20])
    ax.set_ylim([0,20])
    ax.set_zlim([0,20])

    ax.axis('equal')

    plt.title('3D Map Altitudes and Terrain Types')


    #-------------------------------------------------
    # Terminal Condition Plot
    #-------------------------------------------------

    ax = fig.add_subplot(prows, pcols, pnum); pnum=pnum+1


    view = get_view_at( map_tiles_raw, x_end_raw, y_end_raw )

    view_rgb = tiles_to_rgb( view) # np.transpose( np.flipud( view ) ) )

    plt.imshow( view_rgb )

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('Terminal View at {} {}'.format(y_end_raw, map_tiles_raw.shape[0]-x_end_raw-1 ))


    #-------------------------------------------------
    # Time-X plot
    #-------------------------------------------------

    ax = fig.add_subplot(prows, pcols, pnum); pnum=pnum+1

    plt.plot(T,Xd,'r-')
    plt.plot(T,Xd,'r.')

    plt.xlabel('Time')
    plt.ylabel('X')

    plt.title('Time-X Plot')


    #-------------------------------------------------
    # Time-Y plot
    #-------------------------------------------------

    ax = fig.add_subplot(prows, pcols, pnum); pnum=pnum+1

    plt.plot(T,Yd,'r-',label="Drone path")
    plt.plot(T,Yd,'r.',label="Drone points")
    plt.xlabel('Time')
    plt.ylabel('Y')

    plt.title('Time-Y Plot')


    #-------------------------------------------------
    # Time-Altitude plot
    #-------------------------------------------------

    ax = fig.add_subplot(prows, pcols, pnum); pnum=pnum+1

    plt.plot(T,Zd,'r-',label="Drone path")
    plt.plot(T,Zd,'r.',label="Drone points")
    plt.xlabel('Time')
    plt.ylabel('Y')

    plt.title('Time-Altitude Plot')


    #-------------------------------------------------
    # Extras
    #-------------------------------------------------

    annotated_title = title + " Outcome: {} ".format( "Crashed" if crashed else ("Stuck" if stuck else "Dropped OK"))
    plt.suptitle(annotated_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




def plot_map(title="Trajectory", map_name="box_canyon"):

    """Given a pandas dataframe of events from a training run covering multiple episodes,
       and a desired episode number,
       plots the trajectory of the episode in various ways to provide an understanding of the drone behavior.

       :param title - title for top of plots
       :param map_name - map to use as background for the visualizations and feature generation

       ToDo: This routine should be broken up to make the individual graphs usable.
             We should probably standardize the interface on something close to the raw data so
             that routines can be easily applied.
       """


    # Background map data

    map_tiles_raw     = maps.name_to_map[map_name]
    map_altitudes_raw = tiles_to_altitudes(map_tiles_raw)


    # Flip vertically so that imshow will render this the way we expect on the screen

    map_tiles_flip     = np.flipud(map_tiles_raw)  # np.transpose( np.flipud(map_tiles_raw)     )
    map_altitudes_flip = np.flipud(map_altitudes_raw)


    # Flip and transpose so that 3D Bar chart shows expected landscape

    map_tiles_ftp     =  np.transpose( np.flipud(map_tiles_raw)     )
    map_altitudes_ftp = np.transpose( np.flipud(map_altitudes_raw) )





    #-------------------------------------------------
    # Setup Plotting
    #-------------------------------------------------

    fig = plt.figure( figsize=(16,8) )

    prows = 1 # Number of rows in plotted graphs
    pcols = 1 # Number of columns in plotted graphs

    pnum=1


    #-------------------------------------------------
    # 2D Tile Type Map
    #-------------------------------------------------

    ax = fig.add_subplot(prows, pcols, pnum); pnum=pnum+1

    img = tiles_to_rgb(map_tiles_flip)

    plt.imshow(img)


    plt.xlim( [0,20])
    plt.ylim( [0,20])
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('Map')

    plt.show()

if __name__ == "__main__":
    print("Plotting map")
    plot_map("Map","nixel_sample")