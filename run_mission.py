
import socket

#import actr
import uuid
import subprocess
import pickle
import itertools

import bad_navigation


def ListToFormattedString(alist):
    # Each item is right-adjusted, width=3
    # modified from: https: // stackoverflow.com / questions / 7568627 / using - python - string - formatting - with-lists
    formatted_list = ["'{}'" if isinstance(i, str) else "{}" for i in alist]
    # print(formatted_list)
    s = '(' + ','.join(formatted_list) + ')'
    return s.format(*alist)

#TODO
#NOTE: drone going to 90, 340 for [110, 70] -- WHY???
#
pilot_name = "Qualitative_analysis_near_one_tree"
version_number = 2
custom = True
uuids = []
hiker_positions_x = [70, 90, 110, 130, 150, 170, 190]
hiker_positions_y = [50, 70, 90, 110]

combinations = list(itertools.product(hiker_positions_x,hiker_positions_y))
#sed -i 3s:.*:"  <hiker_position>190, 110</hiker_position>": /cogle/cogle-mavsim/cogle_mavsim/assets/godiland_nav_v0.xml

#v1
# combinations = [[100,350],[100,450],[100,150],[384,319],[270,50],[390,50],[410,50],[430,50],[230,70],[270,70]]#,[350,90],[430,110]]
#
# combinations = [[100,350],[100,450],[100,150],[384,319],[270,50],[390,50],[410,50],[430,50],[230,70],[270,70],
#                 [70,50],[90,50],[110,50],[130,50],[150,50],[170,50],[190,50],[70,70],[90,70],[110,70]]#, #v1
                #[350,90],[410,90],[430,110]] #v1
#v2
# combinations = [[70,50],[90,50],[110,50],[130,50],[150,50],[170,50],[190,50],[70,70],[90,70],[110,70]]

#v5
#combinations = [[261,269],[251,269],[243,271],[247,279],[287,263],[288,269]]#[[278,267],[284,277],[268,277],[261,269],[251,269],[243,271],[247,279],[287,263],[288,269]]
#,[130,70],[150,70],[170,70],[190,70],[70,90],[90,90],[110,90], #v2
                # [130,90],[150,90],[170,90],[70,110],[90,110],[110,110],[130,110],[150,110],[170,110], #v2
                # [50,150],[50,250],[100,250],[100,50],  #v2
                # [278,267],[278,273],[284,277],[268,277],[261,269],[251,269],[243,271],[247,279],[287,263], #v5
                # [288,269],[327,264],[389,381],[417,304],[452,312],[380,268],[411,271],[418,277],[431,282]] #v5

#custom combinations
combinations = [[52,473]]
path_coordinates =[[70,481]]

#last_hiker_x = 25
#last_hiker_y = 33
#need to edit the xml file to those values: 25, 33

for combination in combinations:



    print(path_coordinates)

    for i in range(2):

        mission_uuid = uuid.uuid4().hex
        session_uuid = uuid.uuid4().hex
        instance_uuid = uuid.uuid4().hex
        uuids.append({'mission':mission_uuid,'session':session_uuid,'instance':instance_uuid})
        #print("uuid", auuid)
        #ignore
        msgs = [['SIM', 'INSTANCE', 'instance_{}'.format(instance_uuid)],
                ['SIM', 'SESSION', 'session_{}'.format(session_uuid)],
                ['SIM', 'PILOT', '{}_v{}'.format(pilot_name,version_number)],
                ['SIM', 'CLOSE', 'unclosed'],['SIM', 'NEW', 'godiland-base', mission_uuid], ['FLIGHT', 'ARM'],
                ['FLIGHT', 'MS_LOAD_PAYLOAD', 0, 'food'],['FLIGHT', 'MS_LOAD_PAYLOAD', 1, 'medicine'],
                ['FLIGHT', 'MS_LOAD_PAYLOAD', 2, 'communication'],['FLIGHT', 'MS_LOAD_PAYLOAD', 3, 'food'],
                ['FLIGHT', 'AUTO_TAKEOFF'],['FLIGHT', 'MS_NO_ACTION'],['FLIGHT', 'MS_NO_ACTION'],
                ['FLIGHT', 'MS_NO_ACTION'],['FLIGHT', 'MS_NO_ACTION'],['FLIGHT', 'MS_NO_ACTION'],
                ['FLIGHT', 'MS_NO_ACTION'],['FLIGHT', 'MS_NO_ACTION'],['FLIGHT', 'MS_NO_ACTION'],
                ['FLIGHT', 'FLY_TO', 485, 29, 3],['FLIGHT', 'MS_NO_ACTION'],['FLIGHT', 'MS_NO_ACTION'],
                ['FLIGHT', 'MS_NO_ACTION'],['FLIGHT', 'MS_NO_ACTION'],['FLIGHT', 'MS_NO_ACTION']]



        mavsim_server = ('127.0.0.1', 32786)
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        for msg in msgs:
            msg = ListToFormattedString(msg)
            print("sending", msg)
            sent = send_sock.sendto(msg.encode('utf-8'),mavsim_server)
            data, server = send_sock.recvfrom(1024)
            print(data.decode('utf-8'))

        print("take-off procedure complete.")

        #if Simon's xml file has coordinates, fly to them
        #path_coordinates = [[30,483]]
        bad_navigation.fly_path(coordinates=path_coordinates)



        #First edit the xml file
        #sed_command = "0,/{}/s/{}/{}/".format(last_hiker_x,last_hiker_x,x)
        #print(sed_command)
        #last_hiker_x = x
        #subprocess.run(["docker", "exec", "q-agent", "sed", "-i", sed_command, "/cogle/cogle-mavsim/cogle_mavsim/assets/godiland_nav_v0.xml"])
        #Do the same for y


        #sed - i
        #'0,/150/s/150/170/' / cogle / cogle - mavsim / cogle_mavsim / assets / godiland_nav_v0.xml

        print("starting the q-learner.")



        subprocess.run(["docker", "exec", "q-learner-container33", "python3", "main.py", "--env-id", "apl-nav-godiland-v{}".format(version_number), "--drop_payload_agent",
             "--qfunction", "./q_functions/multi_redrop-v{}.qf".format(version_number)])

        print("DONE.")
        print("reseting.")
        msg = "('{}', '{}', '{}')".format('SIM', 'CLOSE', mission_uuid)
        #print(sending, msg)
        sent = send_sock.sendto(msg.encode('utf-8'),mavsim_server)
        data, server = send_sock.recvfrom(1024)
        print(data.decode('utf-8'))

        print("compelete.")
    print(uuids)

    file_name = "{}-{}.p".format(combination[0],combination[1])

    with open(file_name, 'wb') as handle:
        pickle.dump(uuids, handle)


    print("uuids pickled as", file_name)
    uuids = []
print("all done")
#Todo something needs killing