import socket
import threading
import yaml
import time



class MavsimHandler():
    def __init__(self):
        self.mavsim_server = ('127.0.0.1', 32786)
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sent = self.send_sock.sendto(b'(\'TELEMETRY\', \'ADD_LISTENER\', \'docker.for.mac.localhost\', 9048, 0)',
                                self.mavsim_server)

        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.state_address = ('127.0.0.1', 9048)
        self.state_socket.bind(self.state_address)

        self.mavsim_state = {}

        #start the thread
        #self.stateThread =

    def ListToFormattedString(self,alist):
        # Each item is right-adjusted, width=3
        # modified from: https: // stackoverflow.com / questions / 7568627 / using - python - string - formatting - with-lists
        formatted_list = ["'{}'" if isinstance(i, str) else "{}" for i in alist]
        # print(formatted_list)
        s = '(' + ','.join(formatted_list) + ')'
        return s.format(*alist)

    def drop_package(self):
        msg = ['FLIGHT', 'MS_DROP_PAYLOAD', 1]
        msg = self.ListToFormattedString(msg)
        print("msg", msg)
        sent = self.send_sock.sendto(msg.encode('utf-8'), self.mavsim_server)
        time.sleep(0.1)

    def head_to(self,heading,altitude):
        msg = ['FLIGHT','HEAD_TO',heading,1,altitude]
        msg = self.ListToFormattedString(msg)
        print("msg",msg)
        sent = self.send_sock.sendto(msg.encode('utf-8'), self.mavsim_server)
        time.sleep(0.1)

    def fly_path(self,coordinates=[],altitude=3):
        # return 0
        # while not self.mavsim_state:
        #     print("da state is", self.mavsim_state)
        #     time.sleep(0.0001)
        if coordinates:
            #for coordinate in coordinates:
            #print("FLYING TO", coordinates[0],coordinates[1])
            msg = ['FLIGHT', 'FLY_TO', coordinates[1]-1, coordinates[0]-1, altitude]
            msg = self.ListToFormattedString(msg)
            sent = self.send_sock.sendto(msg.encode('utf-8'),self.mavsim_server)
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
                sent = self.send_sock.sendto(msg.encode('utf-8'), self.mavsim_server)
                time.sleep(0.2)

                    # for msg in msgs:
                    #     msg = ListToFormattedString(msg)
                    #     print("sending", msg)
                    #     sent = send_sock.sendto(msg.encode('utf-8'), mavsim_server)
                    #     data, server = send_sock.recvfrom(1024)
                    #     print(data.decode('utf-8'))
        return 1




    def read_state(self):
        while 1:
            data, add = self.state_socket.recvfrom(1024)
            data = data.decode('utf-8')
            #global mavsim_state
            #print(data)
            #print ("da state", self.mavsim_state)
            self.mavsim_state[data[:data.find('{')-1]] = yaml.load(data[data.find('{'):])
            #if 'GLOBAL_POSITION_INT' in state:
            #    latitude = state['GLOBAL_POSITION_INT']['vy']
            #    longitude = state['GLOBAL_POSITION_INT']['vx']
            #    print("lat,lon",latitude, longitude)
            #    #get sensor grid
            #    #sock.sendto(b'(\'FLIGHT\', \'MS_QUERY_TERRAIN\', latitude,\'longitude\')', server_address)
            #    msg = "('FLIGHT', 'MS_QUERY_TERRAIN', {}, {})".format(latitude, longitude)
            #    #sock.sendto(msg.encode('utf-8'), server_address)
            #    sock.sendto(data, server_address)

    #stateThread = threading.Thread(target=read_state)
    #stateThread.start()

    #print("")