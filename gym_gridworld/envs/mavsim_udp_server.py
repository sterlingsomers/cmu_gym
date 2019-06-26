import socket
import threading
import yaml
import time



class MavsimUDPHandler():

    """Provides an interface to the MAVSim aircraft simulation over a UDP socket connection.

       Ideally this interface would also manage map offsets and import map tiles that are required."""


    def __init__(self, mavsim_host='127.0.0.1',
                       mavsim_port=14555,
                       rl_agent_host='127.0.0.1',
                       rl_agent_port=9048,
                       run_synchronous =True):

        print("MavsimHandler initializing")
        print("   mavsim_host:   {}  mavsim_port:   {}".format(mavsim_host,mavsim_port))
        print("   rl_agent_host: {}  rl_agent_port: {} ".format(rl_agent_host,rl_agent_port))


        # Save off parameters
        #--------------------

        self.run_synchronous = run_synchronous

        self.mavsim_host = mavsim_host
        self.mavsim_port = mavsim_port

        self.rl_agent_host = rl_agent_host
        self.rl_agent_port = rl_agent_port

        self.receive_state_socket_buffer_size = 1024

        self.alt_to_z_dict = { 0:0, 35:1, 400:2, 999:3 }



        # Key state variables
        #--------------------

        self.mavsim_server_info = (mavsim_host, mavsim_port)
        
        self.altitude = None
        self.latitude = None
        self.longitude = None
        self.heading = None

        self.received_reply = False


        # Iinitialize local aircraft state proxy
        #---------------------------------------

        self.mavsim_state = {}

        self._setup_receiver_socket()


        # Setup bidirectional communication with MAVSim
        #----------------------------------------------

        self._send_to_mavsim_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.send_to_mavsim(
            "('TELEMETRY', 'ADD_LISTENER', '{}', {}, 0)".format(  self.rl_agent_host, self.rl_agent_port  ))

        print("Registered telemetry request with MAVSim")


        # Arm and launch drone
        #---------------------

        self._get_drone_flying()



    def _setup_receiver_socket(self):


        # Setup socket to receive state from MAVSim
        #------------------------------------------

        self.receive_state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:  # Only some OS use this attribute
            self.receive_state_socket.SO_REUSEPORT
        except AttributeError:
            print("INFO: This platform does NOT support socket.SO_REUSEPORT")
        else:
            self.receive_state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)


        self.state_address = (self.rl_agent_host, self.rl_agent_port)
        self.receive_state_socket.bind(self.state_address)

        print("Starting MAVSim receiver socket thread")
        self.stateThread = threading.Thread(target=self.read_state)
        self.stateThread.start()

        print("Thread async call complete")


    def send_to_mavsim(self,msg_string):

        print("MAVSimHandler.send_to_mavsim msg_string={}".format(msg_string))

        self._send_to_mavsim_socket.sendto(bytes(msg_string, 'utf-8'), self.mavsim_server_info)

        #time.sleep(2)




    def _get_drone_flying(self):

        print("Setting up scenario, arming and launching drone")

        # If there was a previous simulation, this will stop it so that we can start a new one

        self.send_to_mavsim( "('SIM','CLOSE','Closing previous scenario so we can start a new one')")

        self.send_to_mavsim( "('SIM','NEW','kingdom-base-A-7','1234')" )
        self._wait_for_reply()

        self.send_to_mavsim("('FLIGHT','ARM')")
        self._wait_for_reply()

        self.send_to_mavsim("('FLIGHT','AUTO_TAKEOFF',2,15)")
        self._wait_for_reply()

        for i in range(10):
            print("Running flight MS_NO_ACTION to allow takeoff to complete")
            self.no_op()

        print("As far as we know, drone is launched")


    def _wait_for_reply(self):

        if self.run_synchronous:
            while not self.received_reply:
                pass


    def no_op(self):

        self.received_reply=False

        self.send_to_mavsim("('FLIGHT','MS_NO_ACTION')")

        self._wait_for_reply()


    def ListToFormattedString(self,alist):
        # Each item is right-adjusted, width=3
        # modified from: https: // stackoverflow.com / questions / 7568627 / using - python - string - formatting - with-lists
        formatted_list = ["'{}'" if isinstance(i, str) else "{}" for i in alist]
        # print(formatted_list)
        s = '(' + ','.join(formatted_list) + ')'
        return s.format(*alist)


    def drop_package(self):

        self.received_reply=False
        msg = ['FLIGHT', 'MS_DROP_PAYLOAD', 1]
        msg = self.ListToFormattedString(msg)
        print("msg", msg)
        sent = self.send_to_mavsim(msg)
        time.sleep(0.1)

        self._wait_for_reply()


    def head_to(self,heading,altitude):

        self.received_reply=False
        msg = ['FLIGHT','HEAD_TO',heading,1,altitude]
        msg = self.ListToFormattedString(msg)
        print("msg",msg)
        sent = self.send_to_mavsim(msg)

        self._wait_for_reply()


    def fly_path(self,coordinates=[],altitude=3):


        print("WARNING: fly_path is disabled:")
        return 0

        self.received_reply=False

        # return 0
        # while not self.mavsim_state:
        #     print("da state is", self.mavsim_state)
        #     time.sleep(0.0001)
        if coordinates:
            #for coordinate in coordinates:
            #print("FLYING TO", coordinates[0],coordinates[1])
            msg = ['FLIGHT', 'FLY_TO', coordinates[1]-1, coordinates[0]-1, altitude]
            msg = self.ListToFormattedString(msg)
            sent = self.send_to_mavsim(msg)
            time.sleep(0.1)
            #while not 'GLOBAL_POSITION_INT' in self.mavsim_state:
             #   time.sleep(0.0001)
                #print("da state dat state", self.mavsim_state)
            self.timestamp = 0
            while 1:
                timestamp = self.mavsim_state['GLOBAL_POSITION_INT']['time_boot_ms']
                if timestamp == self.timestamp:
                    continue
                if (coordinates[0]-1) == int(self.mavsim_state['GLOBAL_POSITION_INT']['vx']) and \
                        (coordinates[1]-1) == int(self.mavsim_state['GLOBAL_POSITION_INT']['vy']):
                    break


                self.timestamp = self.mavsim_state['GLOBAL_POSITION_INT']['time_boot_ms']
                #coordinates[1] != int(self.mavsim_state['GLOBAL_POSITION_INT']['vx']) and int(coordinates[0] != self.mavsim_state['GLOBAL_POSITION_INT']['vy']):
                msg = ['FLIGHT', 'MS_NO_ACTION']
                msg = self.ListToFormattedString(msg)
                sent = self.send_to_mavsim(msg)
                time.sleep(0.2)

                    # for msg in msgs:
                    #     msg = ListToFormattedString(msg)
                    #     print("sending", msg)
                    #     sent = send_sock.sendto(msg.encode('utf-8'), mavsim_server)
                    #     data, server = send_sock.recvfrom(1024)
                    #     print(data.decode('utf-8'))

        self._wait_for_reply()

        return 1


    def read_state(self):

        print("MAVSimHandler.read_state starting loop")
        while 1:

            # Assuming all messages are less than receive_state_socket_buffer_size
            # and that we completely read every message and there is only one message in the stream, this works??

            #print("Waiting for data")
            data, add = self.receive_state_socket.recvfrom(self.receive_state_socket_buffer_size)
            data = data.decode('utf-8')

            print("Data Received {}".format(data))
            #print ("da state", self.mavsim_state)

            variable =  data[:data.find('{')-1] # Everything up to the brace - the space after the variable name
            values_string = data[data.find('{'):] # List of attributes after the brace -- ignores closing brace

            self.mavsim_state[variable] = yaml.load(values_string)

            # Parsing gets messed up if there are no arguments, so we just use the original data

            if data == "YOUR_TURN":
                #print("End of turn detected reply = True")
                self.received_reply=True


            if 'GLOBAL_POSITION_INT' in self.mavsim_state:

               self.latitude = self.mavsim_state['GLOBAL_POSITION_INT']['vy']
               self.longitude = self.mavsim_state['GLOBAL_POSITION_INT']['vx']
               self.altitude = self.mavsim_state['GLOBAL_POSITION_INT']['vz']
               self.heading = self.mavsim_state['GLOBAL_POSITION_INT']['hdg']


    def get_drone_position(self):

        return (self.altitude, self.latitude, self.longitude)


    def get_drone_heading(self):

        return self.heading

