import sys
sys.path.append("..")

from mavsim import mavsim
import numpy as np
import matplotlib.pyplot as plt
import random
import os
#from mavsimgym.util import *
from gym_gridworld.envs.create_np_map import create_custom_map
import threading
import ast
import yaml




class MavsimLibHandler:

    """Wraps the MAVSim discrete light weight flight simulator with an OpenAI Gym like API
       and performs caching of maps and translation of coordinates between Mavsim and CMU_Drone.

       This framework assumes that we are going to choose one scenario for our experiments,
       but then sample many small maps from this one scenario.

       Currently assumes a postgres database is around to connect to.
       See the _create_mavsim function for details.
    """

    def __init__(self, params):

        self.params = params

        self.drone_location = np.array([-1,-1,-1])  # Location as discrete 3 element integer array (x,y,alt)
        self.drone_heading = -1

        # We represent unknown waypoint as waypoint == drone_location

        self.mavsim = None
        self.map =    None

        #if self.mavsim !=None:
        #    del self.mavsim
        self._create_mavsim()

        self._load_scenario( self.params['scenario'] )
        self._extract_or_load_global_scenario_map()


    def reset(self, submap_offset, submap_shape):

        """Sets up a new scenario. Calls mavsim to reload map. Setsup a new submap for cmu drone to use.
           This is a hard reset that takes a while to execute. """


        self.submap_offset = np.array(submap_offset)
        self.submap_shape =  np.array(submap_shape)
        self._setup_submap(self.submap_offset,self.submap_shape)

        self.trajectory = []
        self.callback_event = threading.Event()

        #self._arm_drone_to_get_initial_location()
        #self._choose_random_goal_near_drone()
        #self.waypoint_location = self.drone_location
        self.crashed=False

        #self._auto_takeoff()

        return self._create_observation()


    def _setup_submap(self, offset, dimensions):

        x1 = offset[0]
        y1 = offset[1]
        x2 = x1+dimensions[0]
        y2 = y1+dimensions[1]

        np_map = self.map_types[x1:x2,y1:y2]

        self.cmu_map = create_custom_map(np_map,offset=(-1,-1))

        self._command("('FLIGHT','MS_SET_AOI', %d, %d, %d, %d)" % (x1, y1, dimensions[0], dimensions[1]))

        print("   completed submap setup")

    def get_submap(self):

        return self.cmu_map


    def get_drone_position(self):

        """Returns drone position as altitude, x,y"""


        # Convert between coordinates in global 512x512 map and local 20x20 map based on offset

        position = self.drone_location - np.array([ self.submap_offset[0],self.submap_offset[1],0])

        # CMU simulator assumes that altitude is the first coordinate instead of the last
        # CMU simulator wants second coordinate to be the row or Y and third coordinate to be the column or X

        cmu_position = np.array( [ position[2],
                                   position[1],position[0]])
        return cmu_position


    def set_drone_position(self, new_position):

        """Takes a CMU style position  altitude,row,col and sets MAVSIM position """

        # SIM LOAD command assumes arguments X,Y,Altitude

        # CMU simulator assumes that altitude is the first coordinate instead of the last
        # CMU simulator wants second coordinate to be the row or Y and third coordinate to be the column or X
        # So we swap axes before sending it to MAVSIM

        global_position = np.array([new_position[2],new_position[1],new_position[0]]) + np.array([ self.submap_offset[0],self.submap_offset[1],0])

        self.drone_location = global_position
        self.crashed = False
        #print("mavsim_lib_server set_drone_position({})->{}".format(new_position, global_position))
        self._command(
            "('SIM','LOAD', %d, %d, %d, 1, 3, 999999, 'True', 1, ['Food', 'Radio', 'Food', 'Radio'], 1, 'True', 0, '[]', '[]')"  \
            % (global_position[0], global_position[1], global_position[2]))


    def set_hiker_position(self, new_position):

        self._command(("('SIM', 'POSITION_HIKER', %d, %d)" % (new_position[0], new_position[1])))


    def get_drone_heading(self):

        return self.drone_heading


    def _auto_takeoff(self):

        self._command("('FLIGHT','AUTO_TAKEOFF',3,7)")
        for i in range(5):
            self._command("('FLIGHT','MS_NO_ACTION')")

        return self._create_observation()


    def _done(self):

        return self.crashed or np.linalg.norm( self.goal_location - self.drone_location ) < 5


    def wait(self):

        self._command("('FLIGHT','MS_NO_ACTION')")

        return self._create_observation(), self._reward(), self._done()


    def head_to(self, heading, distance, altitude ):

        """Implements a protocol in which the action is represented as a tuple (command,x,y,a).

           Command is either FLY_TO.
           The x and y coordinates given the destination for FLY_TO in terms of map pixel coordiantes.
           The a parameter is altitude in the domain specific integer format from 0 to 3"""

        #remap_heading = [ 8,1,2,3,4,5,6,7]

        #new_heading = remap_heading[ heading-1]


        old_drone_location = self.drone_location

        done = False

        # Executing the following command will trigger a callback which updates location state in a hidden fashion

        self._command( "('FLIGHT','HEAD_TO', {},{},{})".format(heading,distance,altitude))

        #print("mavsim_lib_server head_to old drone location {}  new drone location {}".format(old_drone_location, self.drone_location))



    def global_map_show(self):

        fig = plt.figure()

        plt.suptitle("MAVSim Scenario: {} Drone Start:{:3.0f},{:3.0f}".format(self.scenario_name,
                                                                              self.drone_location[0],self.drone_location[1]))

        plt.subplot(1,2,1)
        plt.imshow(self.map_altitudes)
        plt.title("Altitudes")
        plt.xlabel("x / col")
        plt.ylabel("y / row")

        plt.plot(self.drone_location[0],self.drone_location[1],'r+',markersize=12)
        #plt.plot(self.goal_location[0],self.goal_location[1],'rx',markersize=12)

        plt.subplot(1,2,2)
        plt.imshow(self.map_types,cmap='Dark2')
        plt.title("Tile Types")
        plt.xlabel("x / col")
        plt.ylabel("y / row")

        plt.show()


    def submap_show(self):

        fig = plt.figure()
        plt.imshow( self.get_submap() )
        plt.show()


    def get_global_map(self):

        """:return a 2 layer array (2 x width x height)
           where layer 0 is the altitude map and layer 1 is the tile type map"""

        return self.map


    def plot_trajectory(self):

        trajectory_arr = np.array( self.trajectory )
        print(trajectory_arr.shape)

        plt.plot( trajectory_arr[:,0], trajectory_arr[:,1], 'r-+' )

        for i in range( trajectory_arr.shape[0]):
            plt.text(
                trajectory_arr[i,0]+random.random()*0.5-0.25,
                trajectory_arr[i,1]+random.random()*0.5-0.25,
                "{}".format(i),
                fontsize=8)
        plt.axis("equal")
        plt.show()


    def _create_observation(self):

        return self.drone_location


    def _arm_drone_to_get_initial_location(self):

        self.mavsim.command("('FLIGHT','ARM')")


    def _choose_random_goal_near_drone(self):

        self.goal_location =  self.drone_location + (np.random.randint(0,100,size=(1,3))-50)[0]

        self.goal_location = np.clip( self.goal_location, 0, self.map_altitudes.shape[0])
        self.goal_location[2]=3


    def _create_mavsim(self):

        print()
        print("Creating MavSim Instance")
        print("Note: we currently assume there is a postgres database running on the standard port")
        print("One way to get this is to download a postgres docker and configure it as follows")
        print("docker run --rm --name pg-docker -e POSTGRES_PASSWORD=docker -d -p 5432:5432 -v $HOME/docker/volumes/postgres:/var/lib/postgresql/data postgres")
        print()


        self.mavsim = mavsim.MAVSim(
            verbose = self.params['verbose'],
            quiet   = True,
            nodb    = self.params['nodb'],
            server_ip = '0.0.0.0',
            server_port = 14555,
            instance_name = 'MAVSim',
            session_name = 'Training Mission 1',
            pilot_name   = 'Sally',
            database_url =  self.params['database_url'], #'postgresql://postgres:docker@localhost:5432/apm_missions', # -- need to set this to none otherwise it connects anyway
            telemetry_cb = lambda msg: self._callback(msg),
            sim_op_state = 1 )


    def _load_scenario(self,scenario_name):


        # Make sure any previous scenario is terminated, otherwise new one will not start

        #self.mavsim.command( "('SIM','CLOSE','Closing previous scenario so we can start a new one')")


        # Start new scenario

        print("Overriding scenario name to avoid using nixel spec as filename in disk cache")
        self.scenario_name = 'nixel_test'


        #dna = "['COGLE_0:stubland_1:512_2:512_3:256_4:7_5:24|-0.1426885426044464/Terrain_0:0_1:100_2:0.05_3:0.5_4:0.05_5:0.5_6:0.05_7:0.5_8:0.5_9:0.5_10:0.7_11:0.3_12:0.5_13:0.5_14:True/', '0.36023542284965515/Ocean_0:60/', '-0.43587446212768555/River_0:0.01_1:100/', '-0.3501245081424713/Tree_0:500_1:20.0_2:4.0_3:0.01_4:2.0_5:0.1_6:1.9_7:3.0_8:2.2_9:3.5/', '0.6151155829429626/Airport_0:15.0_1:25_2:35_3:1000_4:[]/', '0.34627288579940796/Building_0:150_1:10.0_2:[]_3:1/', '0.31582069396972656/Road_0:3_1:500/', '-0.061891376972198486/DropPackageMission_0:1_3:Find the hiker last located at (88, 186, 41)_4:Provision the hiker with Food_5:Return and report to Southeast International Airport (SEI) airport_6:Southeast Regional Airport_7:Southeast International Airport_8:0_9:20.0_10:20.0_11:40.0/', '-0.25830233097076416/Stub_0:0.8_1:1.0_2:1.0_3:1.0_4:1.0_5:1.0/']"
        dna = self.params['scenario']

        cmd = "('SIM','NEW',\"{}\",'1234')".format(dna)

        print("Calling mavsim with : {}".format(cmd))

        self.mavsim.command( cmd )



    def _extract_or_load_global_scenario_map(self):

        """first checks to see if we have the global scenario map locally on disk,
           if not, it creates a new map and pulls information from mavsim for each cell.
           The new map will be cached for the future.

           ASSUMES that _load_scenario has been called"""

        #self.map_shape = ( self.mavsim.craft.sim.tmx_height,
        #                   self.mavsim.craft.sim.tmx_width   )

        self.map_shape=(512,512)

        print("mavsim_lib_server requesting map from mavsim")
        print("shape {}".format(self.map_shape))

        if not os.path.isfile("map_"+self.scenario_name+"_types.npy"):

            print("mavsim_lib_server map not found")
            self.map_altitudes = np.zeros(self.map_shape)
            self.map_types = np.zeros(self.map_shape)

            for i in range(self.map_shape[0]):
                print("  {} of {}".format(i,self.map_shape[0]))
                for j in range(self.map_shape[1]):
                    blc = self.mavsim.craft.sim.about_this_coordinate(j,i)
                    self.map_altitudes[i][j] = blc.altitude
                    self.map_types[i][j] = blc.tiletype

                    if self.map_types[i][j]==0:
                        self.map_types[i][j]=2  # For now, set zeros to grass ...


            np.save("map_"+self.scenario_name+"_altitudes.npy",self.map_altitudes)
            np.save("map_"+self.scenario_name+"_types.npy",self.map_types)

        else:
            print("mavsim_lib_server map not found")
            self.map_altitudes = np.load("map_"+self.scenario_name+"_altitudes.npy")
            self.map_types = np.load("map_"+self.scenario_name+"_types.npy")

        self.map = np.stack( (self.map_altitudes, self.map_types ), axis = 2)

        #self.global_map_show()


    def _command(self,command_str,block=False):


        # It turns out the mavsim.command library function is actually synchronous in the sense
        # that it will send out all of its callbacks before returning so we don't need external coordination.

        response = self.mavsim.command(command_str)

        if  self.params['halt_on_error'] and 'ERR' in response:
            raise Exception("mavsim generated an error : "+response)


    def parse_message(self,msg):

        """Takes a callback message from MAVSIM and extracts the 'COMMAND' and a dictionary of attributes and returns it

        :param string :msg
        :return ( command_str,  dictionary_of_attribute_value_pairs ) """


        command_str = "ERROR"
        attributes = {}

        try:
            command_str, _, arg_string = msg.partition(" ")
            #attributes = literal_eval(arg_string)
            # THIS IS NOT SAFE, BUT DICTIONARIES ARE NOT LITERAL COMPLIANT
            attributes = yaml.load(arg_string)
            #attributes = ast.literal_eval(arg_string)

        except Exception as e:
            print("mavsim_lib_server could not parse ",arg_string)
            print("Error ",e)

        return (command_str, attributes)


    def _callback(self, message):

        #print("CALLBACK MSG >> %s" % message)
        command_str, attributes = self.parse_message(message)
        #print("   Command ",command_str)
        #print("   Args", attributes)

        try:

            if command_str=="GLOBAL_POSITION_INT":

                new_location= np.array([attributes['vx'],attributes['vy'],attributes['vz']] )

                self.drone_heading = attributes['hdg']

                #print("mavsim_lib_server updated drone location {} heading {}".format(new_location,self.drone_heading))


                if new_location[0] >= self.submap_offset[0] and \
                   new_location[1] >= self.submap_offset[1] and \
                   new_location[0] <  self.submap_offset[0] + self.submap_shape[0] and \
                   new_location[1] <  self.submap_offset[1] + self.submap_shape[1]:

                   # print("mavsim_lib_server drone inside 20x20 grid {}")

                    self.drone_location = new_location
                else:
                    print("mavsim_lib_server drone location {} outside 20x20 grid {} CRASHED ".format(new_location,self.submap_offset))

                    self.crashed = True

                self.trajectory.append( self.drone_location )



            if command_str == "MS_CRAFT_CRASH":

                self.crashed=True
                print("SET CRASHED TO TRUE")

            if command_str == "YOUR_TURN":
                self.callback_event.set()

        except:
            pass





