import sys
sys.path.append("..")

from mavsim import mavsim
import numpy as np
import matplotlib.pyplot as plt
import random
import os
#from mavsimgym.util import *
from gym_gridworld.envs.create_np_map import create_custom_map
from gym_gridworld.features.feature_value_conversion_text_dict import feature_value_map
import threading
import ast
import yaml




class MavsimLibHandler:

    """Wraps the MAVSim discrete light weight flight simulator with an OpenAI Gym like API
       and performs caching of maps and translation of coordinates between Mavsim and CMU_Drone.

       This framework assumes that we are going to choose one scenario for our experiments,
       but then sample many small maps from this one scenario.
    """

    def __init__(self, params):

        self.params = params
        self.submap_shape = (20,20)

        self.submap_offset_row_col = (-1000,-1000)

        self.drone_location_alt_row_col = np.array([-1000, -1000, -1000])  # Location as discrete 3 element integer array (alt,x,y)
        self.drone_heading = -1

        # We represent unknown waypoint as waypoint == drone_location

        self.mavsim = None
        self.global_map =    None

        #if self.mavsim !=None:
        #    del self.mavsim
        self._create_mavsim()


        self._extract_or_load_global_scenario_map()



    def reset(self, submap_offset_row_col, submap_shape):

        """Sets up a new scenario. Calls mavsim to reload map. Sets up a new submap for cmu drone to use.

           submap_offset assumes CMU style coordinates row,col global coordinates because these reference
           our copy of the MAVSIM map which is stored MAVSIM reality"""

        print("mavsim_lib_server reset submap_offset {} submap_shape {}".format(submap_offset_row_col, submap_shape))

        self.submap_offset_row_col = np.array(submap_offset_row_col)
        self.submap_shape =  np.array(submap_shape)
        self._setup_submap(self.submap_offset_row_col, self.submap_shape)
        self.crashed=False



    def _setup_submap(self, offset_row_col, dimensions_row_col):

        r1 = offset_row_col[0]   # row
        c1 = offset_row_col[1]   # col

        r2 = r1 + dimensions_row_col[0]
        c2 = c1 + dimensions_row_col[1]

        np_map = self.map_type_ids[r1:r2, c1:c2]

        self.cmu_map_dict = create_custom_map( np_map, offset=(r1, c1) )

        # NOT SURE IF WE Need to flip offset row,col to x,y for MAVSIM and also flip dimensions
        # TODO Figure this out??

        self._command("('FLIGHT','MS_SET_AOI', %d, %d, %d, %d)" % (r1, c1, dimensions_row_col[1], dimensions_row_col[0]))

        #print("   completed submap setup")



    def get_drone_position(self):

        """Returns drone position as (altitude,row,col) in local map coordinates"""

        return self.drone_location_alt_row_col


    def set_drone_position(self, new_cmu_position_alt_row_col, new_heading ):

        """Takes a local map CMU style position  (altitude,row,col) and sets MAVSIM position """

        # Cache CMU position in driver
        # MAVSim will not send a GLOBAL POSITION INT message until first step, so we cache this here
        # so that the CMU side can display its position before the first action takes place.

        self.drone_location_alt_row_col = new_cmu_position_alt_row_col

        # Write position through to MAVSIM simulator


        if self.params['verbose']:
            print("mavsim_lib_server set_drone_position new CMU position: {} heading: {} cmu offset: {} {}".format(
                new_cmu_position_alt_row_col,
                new_heading,
                self.submap_offset_row_col[0],
                self.submap_offset_row_col[1]))

        # SIM LOAD command assumes MAVSIM style arguments X,Y,Altitude where X,Y are in global coordinates
        # So we swap CMU row,col axes
        # MAVSim seems to treat offset as row, col

        global_position_x_y_alt =   np.array( [ new_cmu_position_alt_row_col[2] + self.submap_offset_row_col[1],
                                                new_cmu_position_alt_row_col[1] + self.submap_offset_row_col[0],
                                                new_cmu_position_alt_row_col[0] + 0
                                              ] )

        self.drone_heading = new_heading
        self.crashed = False

        if self.params['verbose']:
            print("mavsim_lib_server set_drone_position new MAVSIM position: {} heading: {}".format(
                global_position_x_y_alt, self.drone_heading))

        speed = 1

        self._command(
            "('SIM','LOAD', %d, %d, %d, %d, %d, 999999, 'True', 1, ['Food', 'Radio', 'Food', 'Radio'], 1, 'True', 0, '[]', '[]')"  \
            % (global_position_x_y_alt[0],
               global_position_x_y_alt[1],
               global_position_x_y_alt[2],
               speed,
               new_heading ))


    def set_hiker_position(self, local_position_row_col):

        """CMU assumes local row,col, MAVSIM is global X,Y  """

        self._command(("('SIM', 'POSITION_HIKER', %d, %d)" % (
            local_position_row_col[1] + self.submap_offset_row_col[1],
            local_position_row_col[0] + self.submap_offset_row_col[0])))


    def get_drone_heading(self):

        """Returns the drone headings as an integer in [1,8] with 1=NORTH, 2=NORTH_EAST, etc."""

        return self.drone_heading


    def head_to(self, heading, distance, altitude ):

        """Causes MAVSim to attempt to fly in direction given by heading for a distance and a specific altitude"""

        # Executing the following command will trigger a callback which updates location state in a hidden fashion

        self._command( "('FLIGHT','HEAD_TO', {},{},{})".format(heading,distance,altitude))

        #print("mavsim_lib_server head_to old drone location {}  new drone location {}".format(old_drone_location, self.drone_location))


    def get_global_map(self):

        """:return a 2 layer array (2 x width x height)
           where layer 0 is the altitude map and layer 1 is the tile type map"""

        return self.global_map


    def _create_mavsim(self):

        print()
        print("Creating MavSim Instance")
        print("Note: we currently assume there is a postgres database running on the standard port")
        print("One way to get this is to download a postgres docker and configure it as follows")
        print("docker run --rm --name pg-docker -e POSTGRES_PASSWORD=docker -d -p 5432:5432 -v $HOME/docker/volumes/postgres:/var/lib/postgresql/data postgres")
        print()


        if 'use_mavsim_instance' in self.params:
            print("mavsim_lib_server.py received a mavsim instance to use")
            self.mavsim= self.params['use_mavsim_instance']
            self.scenario_name='nixel_test'
        else:
            print("mavsim_lib_server.py creating a new mavsim instance")
            self.mavsim = mavsim.MAVSim(
                verbose = self.params['verbose'],
                quiet   = True,
                nodb    = self.params['nodb'],
                server_ip = self.params['server_ip'],
                server_port = 14555,
                instance_name = 'MAVSim',
                session_name = 'Training Mission 1',
                pilot_name   = 'Sally',
                database_url =  self.params['database_url'], #'postgresql://postgres:docker@localhost:5432/apm_missions', # -- need to set this to none otherwise it connects anyway
                telemetry_cb = lambda msg: self._callback(msg),
                sim_op_state = 1 )

            self._load_scenario( self.params['scenario'] )


    def _load_scenario(self,scenario_name):

        # Start new scenario

        print("Overriding scenario name to avoid using nixel spec as filename in disk cache")
        self.scenario_name = 'nixel_test'


        #dna = "['COGLE_0:stubland_1:512_2:512_3:256_4:7_5:24|-0.1426885426044464/Terrain_0:0_1:100_2:0.05_3:0.5_4:0.05_5:0.5_6:0.05_7:0.5_8:0.5_9:0.5_10:0.7_11:0.3_12:0.5_13:0.5_14:True/', '0.36023542284965515/Ocean_0:60/', '-0.43587446212768555/River_0:0.01_1:100/', '-0.3501245081424713/Tree_0:500_1:20.0_2:4.0_3:0.01_4:2.0_5:0.1_6:1.9_7:3.0_8:2.2_9:3.5/', '0.6151155829429626/Airport_0:15.0_1:25_2:35_3:1000_4:[]/', '0.34627288579940796/Building_0:150_1:10.0_2:[]_3:1/', '0.31582069396972656/Road_0:3_1:500/', '-0.061891376972198486/DropPackageMission_0:1_3:Find the hiker last located at (88, 186, 41)_4:Provision the hiker with Food_5:Return and report to Southeast International Airport (SEI) airport_6:Southeast Regional Airport_7:Southeast International Airport_8:0_9:20.0_10:20.0_11:40.0/', '-0.25830233097076416/Stub_0:0.8_1:1.0_2:1.0_3:1.0_4:1.0_5:1.0/']"
        dna = self.params['scenario']

        cmd = "('SIM','NEW',\"{}\",'1234')".format(dna)

        print("Calling mavsim with : {}".format(cmd))

        response = self._command( cmd )

        print("Created new sim with response: {}".format(response))


    def _pull_map_for_location_size(self, row, col, w, h):

        "Takes an offset in CMU global coordinates and returns submap of size w x h"

        altitudes = np.zeros( [w,h])

        #tile_type_ids  = np.zeros( [w,h] )
        # for row in range(w):
        #     for col in range(h):
        #
        #         # MAVSim actually stores the map in the same way???
        #         v = self._command("('FLIGHT', 'MS_QUERY_TERRAIN', %d, %d)" % (row, col))
        #         #print(v)
        #         val = ast.literal_eval(v)
        #
        #         altitudes[row][col] = val[4]
        #         tile_type_name_str = val[5]
        #         #print("mavsim_lib_server _extract_or_load_global_scenario_map looking up feature name >{}<".format(tile_type_name_str))
        #         tile_type_id = feature_value_map[ tile_type_name_str ]['val']
        #
        #         tile_type_ids[row][col] = tile_type_id



        # Newer API here:

        print("mavsim_lib_server._pull_map_for_location_size sending MS_QUERY_SLICE to MAVSim offset r:{} c:{} h:{} w:{}".format(row,col,h,w))
        response_str = self._command("('FLIGHT', 'MS_QUERY_SLICE', %d, %d, %d, %d)" % ( row, col, w, h) )

        #print(response_str)
        # ('ACK', 'MS_QUERY_SLICE', '[ [ 24, ...], ... ]' )
        response_tuple_of_strings = ast.literal_eval(response_str)
        array_list = ast.literal_eval(response_tuple_of_strings[2])
        tile_type_ids = np.array( array_list )

        #print(tile_type_ids)


        return (altitudes,tile_type_ids)


    def _extract_or_load_global_scenario_map(self):

        """first checks to see if we have the global scenario map locally on disk,
           if not, it creates a new map and pulls information from mavsim for each cell.
           The new map will be cached for the future.

           ASSUMES that _load_scenario has been called"""

        #self.map_shape = ( self.mavsim.craft.sim.tmx_height,
        #                   self.mavsim.craft.sim.tmx_width   )

        self.map_shape=(512,512)

        print("mavsim_lib_server looking for map ")
        print("mavsim_lib_server WARNING MAP shape hard wired to {}".format(self.map_shape))

        if not os.path.isfile("map_"+self.scenario_name+"_types.npy"):

            print("mavsim_lib_server map cache not found, pulling new map from MAVSIM")

            self.map_altitudes,self.map_type_ids = self._pull_map_for_location_size(0, 0, self.map_shape[0], self.map_shape[1])


            # Hacked code that pulls direct from MAVSIM

            #for i in range(self.map_shape[0]):
            #    print("  {} of {}".format(i,self.map_shape[0]))
            #    for j in range(self.map_shape[1]):
            #        blc = self.mavsim.craft.sim.about_this_coordinate(j,i)
            #        self.map_altitudes[i][j] = blc.altitude
            #        self.map_types[i][j] = blc.tiletype
            #
            #        if self.map_types[i][j]==0:
            #            self.map_types[i][j]=2  # For now, set zeros to grass ...


            np.save("map_"+self.scenario_name+"_altitudes.npy",self.map_altitudes)
            np.save("map_" + self.scenario_name +"_types.npy", self.map_type_ids)

        else:
            print("mavsim_lib_server map cache found, loading from disk")

            self.map_altitudes = np.load("map_"+self.scenario_name+"_altitudes.npy")
            self.map_type_ids = np.load("map_" + self.scenario_name + "_types.npy")

        self.global_map = np.stack((self.map_altitudes, self.map_type_ids), axis = 2)

        if self.params['show_global_map']:
            self.global_map_show()


    def _command(self,command_str,block=False):


        # It turns out the mavsim.command library function is actually synchronous in the sense
        # that it will send out all of its callbacks before returning so we don't need external coordination.

        response = self.mavsim.command(command_str)

        if self.params['verbose']:
            print("mavsim_lib_server response:{}".format(response))

        if  self.params['halt_on_error'] and "'ERR'" in response:
            raise Exception("mavsim generated an error : "+response)

        return response


    def parse_message(self,msg):

        """Takes a callback message from MAVSIM and extracts the 'COMMAND' and a dictionary of attributes and returns it

        :param string :msg
        :return ( command_str,  dictionary_of_attribute_value_pairs ) """


        command_str = "mavsim_lib_server PARSE ERROR"
        attributes = {}

        try:
            command_str, _, arg_string = msg.partition(" ")
            #attributes = literal_eval(arg_string)
            # THIS IS NOT SAFE, BUT DICTIONARIES ARE NOT LITERAL COMPLIANT

            print("mavsim_lib_server.parse_message command:{} args:{}".format(command_str, arg_string))
            if not arg_string.isspace() and len(arg_string)>0 :
                attributes = yaml.load(arg_string)
            else:
                attributes={}
            #attributes = ast.literal_eval(arg_string)

        except Exception as e:
            print("mavsim_lib_server could not parse ",arg_string)
            print("Error ",e)

        return (command_str, attributes)


    def _callback(self, message):

        if self.params['verbose']:
            print("mavsim_lib_server _callback message = >>%s<<" % message)

        command_str, attributes = self.parse_message(message)
        #print("   Command ",command_str)
        #print("   Args", attributes)


        #try:

        if command_str=="GLOBAL_POSITION_INT":

            location_x_y_z = np.array([attributes['vx'],attributes['vy'],attributes['vz']] )
            self.drone_heading = attributes['hdg']

            cmu_loc_alt_row_col = [ location_x_y_z[2],
                                    location_x_y_z[1] - self.submap_offset_row_col[0],
                                    location_x_y_z[0] - self.submap_offset_row_col[1]   ]

            if self.params['verbose']:
                print("mavsim_lib_server drone location mavsim global x y z {} heading {}".format(location_x_y_z,self.drone_heading))
                print("mavsim_lib_server drone location cmu    local  a r c {} ".format(cmu_loc_alt_row_col))


            # Need to compare MAVSIM X against CMU column
            # But MAVSim offset and CMU offsets are both row,col

            if      0 <= cmu_loc_alt_row_col[1]  and  cmu_loc_alt_row_col[1] < self.submap_shape[0] \
               and  0 <= cmu_loc_alt_row_col[2]  and  cmu_loc_alt_row_col[2] < self.submap_shape[1]   :

               # print("mavsim_lib_server drone inside 20x20 grid {}")

                self.drone_location_alt_row_col = cmu_loc_alt_row_col

            else:
                print("mavsim_lib_server drone location global row: {} col: {} alt:{} outside 20x20 grid {} CRASHED ".format(
                    location_x_y_z[1],
                    location_x_y_z[0],
                    location_x_y_z[2],
                    self.submap_offset_row_col))

                self.crashed = True

        if False and command_str=="INITIAL_STATE":

            location_x_y_z = np.array([attributes['x'],attributes['y'],attributes['z']] )
            self.drone_heading = attributes['h']

            cmu_loc_alt_row_col = [ location_x_y_z[2],
                                    location_x_y_z[1] - self.submap_offset_row_col[0],
                                    location_x_y_z[0] - self.submap_offset_row_col[1]   ]

            if self.params['verbose']:
                print("mavsim_lib_server drone location mavsim global x y z {} heading {}".format(location_x_y_z,self.drone_heading))
                print("mavsim_lib_server drone location cmu    local  a r c {} ".format(cmu_loc_alt_row_col))


            # Need to compare MAVSIM X against CMU column
            # But MAVSim offset and CMU offsets are both row,col

            if      0 <= cmu_loc_alt_row_col[1]  and  cmu_loc_alt_row_col[1] < self.submap_shape[0] \
                    and  0 <= cmu_loc_alt_row_col[2]  and  cmu_loc_alt_row_col[2] < self.submap_shape[1]   :

                # print("mavsim_lib_server drone inside 20x20 grid {}")

                self.drone_location_alt_row_col = cmu_loc_alt_row_col

            else:
                print("mavsim_lib_server INITIAL_STATE drone location global row: {} col: {} alt:{} outside 20x20 grid {} CRASHED ".format(
                    location_x_y_z[1],
                    location_x_y_z[0],
                    location_x_y_z[2],
                    self.submap_offset_row_col))

                self.crashed = True

        if command_str == "MS_CRAFT_CRASH":

            self.crashed=True
            if self.params['verbose']:
               print("mavsim_lib_server callback received CRASH")

        # If we don't catch this, it goes back up into mavsim ... here we just print something
        #except Exception as e:
        #    print("mavsim_lib_server._callback suffered internal error: ")
        #    print("   Error ",e)

    def get_submap(self):

        return self.cmu_map_dict


    def global_map_show(self):

        """Displays the map in a window for visual inspection during debugging"""

        fig = plt.figure()

        plt.suptitle("MAVSim Scenario: {} Drone Row:{:3.0f} Col:{:3.0f}".format(self.scenario_name,
                                                                                self.drone_location_alt_row_col[0],
                                                                                self.drone_location_alt_row_col[1]))

        plt.subplot(1,2,1)
        plt.imshow(self.map_altitudes)
        plt.title("Altitudes")
        plt.xlabel("x / col")
        plt.ylabel("y / row")


        plt.subplot(1,2,2)
        plt.imshow(self.map_type_ids, cmap='Dark2')
        plt.title("Tile Types")
        plt.xlabel("x / col")
        plt.ylabel("y / row")

        plt.plot( self.drone_location_alt_row_col[0], self.drone_location_alt_row_col[1], 'r+', markersize=12)

        plt.show()


    def submap_show(self):

        fig = plt.figure()
        plt.imshow( self.get_submap() )
        plt.show()



    def _arm_drone_to_get_initial_location(self):

        self._command("('FLIGHT','ARM')")


    def _choose_random_goal_near_drone(self):

        self.goal_location = self.drone_location_alt_row_col + (np.random.randint(0, 100, size=(1, 3)) - 50)[0]

        self.goal_location = np.clip( self.goal_location, 0, self.map_altitudes.shape[0])
        self.goal_location[2]=3


    def _auto_takeoff(self):

        self._command("('FLIGHT','AUTO_TAKEOFF',3,7)")
        for i in range(5):
            self._command("('FLIGHT','MS_NO_ACTION')")


    def wait(self):

        self._command("('FLIGHT','MS_NO_ACTION')")

        return self._create_observation(), self._reward(), self._done()



