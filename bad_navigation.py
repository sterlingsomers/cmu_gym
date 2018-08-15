import socket
import threading
import yaml
import time

mavsim_server = ('127.0.0.1', 32786)
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sent = send_sock.sendto(b'(\'TELEMETRY\', \'ADD_LISTENER\', \'docker.for.mac.localhost\', 9048, 0)', mavsim_server)

state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
state_address = ('127.0.0.1', 9048)
state_socket.bind(state_address)

state = {}



def ListToFormattedString(alist):
    # Each item is right-adjusted, width=3
    # modified from: https: // stackoverflow.com / questions / 7568627 / using - python - string - formatting - with-lists
    formatted_list = ["'{}'" if isinstance(i, str) else "{}" for i in alist]
    # print(formatted_list)
    s = '(' + ','.join(formatted_list) + ')'
    return s.format(*alist)

def fly_path(coordinates=[]):
    if coordinates:
        for coordinate in coordinates:
            print("FLYING TO", coordinate[0],coordinate[1])
            msg = ['FLIGHT', 'FLY_TO', coordinate[1], coordinate[0], 3]
            msg = ListToFormattedString(msg)
            sent = send_sock.sendto(msg.encode('utf-8'),mavsim_server)
            steps = 0
            while coordinate[0] != int(state['GLOBAL_POSITION_INT']['vx']) -1  or coordinate[1] != int(state['GLOBAL_POSITION_INT']['vy']) - 1:
                steps += 1
                if steps > 500:
                    break
                msg = ['FLIGHT', 'MS_NO_ACTION']
                msg = ListToFormattedString(msg)
                sent = send_sock.sendto(msg.encode('utf-8'), mavsim_server)
                time.sleep(0.3)
                # print(coordinate[0], coordinate[1], state['GLOBAL_POSITION_INT']['vx'],
                #       state['GLOBAL_POSITION_INT']['vy'])
                # print(coordinate[0] != int(state['GLOBAL_POSITION_INT']['vx']) and coordinate[1] != int(
                #     state['GLOBAL_POSITION_INT']['vy']))

                # for msg in msgs:
                #     msg = ListToFormattedString(msg)
                #     print("sending", msg)
                #     sent = send_sock.sendto(msg.encode('utf-8'), mavsim_server)
                #     data, server = send_sock.recvfrom(1024)
                #     print(data.decode('utf-8'))




def read_state():
    while 1:
        data, add = state_socket.recvfrom(1024)
        data = data.decode('utf-8')
        global state
        #print(data)
        state[data[:data.find('{')-1]] = yaml.load(data[data.find('{'):])
        #if 'GLOBAL_POSITION_INT' in state:
        #    latitude = state['GLOBAL_POSITION_INT']['vy']
        #    longitude = state['GLOBAL_POSITION_INT']['vx']
        #    print("lat,lon",latitude, longitude)
        #    #get sensor grid
        #    #sock.sendto(b'(\'FLIGHT\', \'MS_QUERY_TERRAIN\', latitude,\'longitude\')', server_address)
        #    msg = "('FLIGHT', 'MS_QUERY_TERRAIN', {}, {})".format(latitude, longitude)
        #    #sock.sendto(msg.encode('utf-8'), server_address)
        #    sock.sendto(data, server_address)

stateThread = threading.Thread(target=read_state)
stateThread.start()

print("")