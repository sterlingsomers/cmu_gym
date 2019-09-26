import random
import math
import os
import shutil
import sys
from datetime import datetime
from time import sleep
import pickle
import pygame, time
import numpy as np

from absl import flags

import gym_gridworld.envs.gridworld_env as GridWorld

random.seed(42)

FLAGS = flags.FLAGS
# flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
# flags.DEFINE_bool("Save", False, "Whether to save the collected data of the agents.")
# flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
# flags.DEFINE_integer("step_mul", 100, "Game steps per agent step.")
# flags.DEFINE_integer("n_envs", 20, "Number of environments to run in parallel")
flags.DEFINE_integer("episodes", 1, "Number of complete episodes")
# flags.DEFINE_integer("n_steps_per_batch", 32,
#     "Number of steps per batch, if None use 8 for a2c and 128 for ppo")  # (MINE) TIMESTEPS HERE!!! You need them cauz you dont want to run till it finds the beacon especially at first episodes - will take forever
# flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
# flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
# flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
# flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
# flags.DEFINE_string("model_name", "Drop_Agent", "Name for checkpoints and tensorboard summaries") # Last best Drop_2020_new_terrain_new_reward_2
# flags.DEFINE_integer("K_batches", 10000, # Batch is like a training epoch!
#     "Number of training batches to run in thousands, use -1 to run forever") #(MINE) not for now
# flags.DEFINE_string("map_name", "DefeatRoaches", "Name of a map to use.")
# flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
# flags.DEFINE_boolean("training", False,
#     "if should train the model, if false then save only episode score summaries")
# flags.DEFINE_enum("if_output_exists", "overwrite", ["fail", "overwrite", "continue"],
#     "What to do if summary and model output exists, only for training, is ignored if notraining")
#
# flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")



#human subject flags
flags.DEFINE_string("participant", 'Test', "The participants name")
flags.DEFINE_string("map", 'flatland', "river, canyon, v-river, treeline, small-canyon, flatland")
flags.DEFINE_integer("configuration", 0, "0,1, or 2")



# flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

#flags.DEFINE_string("map", None, "Name of a map to use.")


FLAGS(sys.argv)

###some global variables that should be useful in multiple spots
action_slots = ['left_down','diagonal_left_down','center_down','diagonal_right_down','right_down',
                   'left_level','diagonal_left_level','center_level','diagonal_right_level','right_level',
                   'left_up','diagonal_left_up','center_up','diagonal_right_up','right_up', 'drop']
#excluds entropy and FC
observation_slots = ['hiker_left', 'hiker_diagonal_left', 'hiker_center', 'hiker_diagonal_right', 'hiker_right',
              'ego_left', 'ego_diagonal_left', 'ego_center', 'ego_diagonal_right', 'ego_right',
              'altitude', 'distance_to_hiker']

combos_to_actions = {'left_down':0,'diagonal_left_down':1,'center_down':2,
                     'diagonal_right_down':3,'right_down':4,
                     'left_level':5,'diagonal_left_level':6,'center_level':7,
                     'diagonal_right_level':8,'right_level':9,
                     'left_up':10,'diagonal_left_up':11,'center_up':12,
                     'diagonal_right_up':13,'right_up':14,'drop':15}

action_to_category_map = {
    0: ['left','down'],
    1: ['diagonal_left','down'],
    2: ['center', 'down'],
    3: ['diagonal_right','down'],
    4: ['right', 'down'],
    5: ['left','level'],
    6: ['diagonal_left','level'],
    7: ['center', 'level'],
    8: ['diagonal_right','level'],
    9: ['right', 'level'],
    10: ['left','up'],
    11: ['diagonal_left','up'],
    12: ['center', 'up'],
    13: ['diagonal_right','up'],
    14: ['right', 'up'],
    15: ['drop']
}

possible_actions_map = {
        1: [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]],
        2: [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]],
        3: [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]],
        4: [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]],
        5: [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1]],
        6: [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
        7: [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]],
        8: [[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

    }

FC_distances = []



def convert_dict_chunk_to_vector(chunk,keys):
    '''cannot just take the values because it does not preserve order. this function makes a list of values, all in the same order'''
    values = [chunk[slot] for slot in keys]
    return values

def convert_vector_to_dict_chunk(vector,keys):
    '''len of vector must equal len of keys'''
    return {slot:value for slot,value in zip(keys,vector) if not slot == 'action_probs'}

def strip_slots(chunk, slots):
    '''strips actions away, leaving only observations'''
    for action in slots:
        del chunk[action]
    return chunk

def unpack_chunks(chunks):
    temp_chunks = []
    all_chunks = []
    for action in chunks:
        for chunk in chunks[action]:
            chunk = chunk[:-4]
            achunk = {}
            for i in range(1,len(chunk),2):
                slot,value = chunk[i][0], chunk[i][1]
                achunk[slot] = value
            for possible_action in action_slots:
                achunk[possible_action] = int(possible_action == action)

            # print('test')
            all_chunks.append(achunk)
    return all_chunks

def distance_to_hiker(drone_position,hiker_position):
    distance = np.linalg.norm(drone_position-hiker_position)
    return distance


def altitudes_from_egocentric_slice(egocentric_slice):
    alts = np.count_nonzero(egocentric_slice, axis=0)
    alts = [int(x) for x in alts]
    return alts


def egocentric_representation(drone_position, drone_heading, volume):
    ego_slice = np.zeros((5,5))
    column_number = 0
    for xy in possible_actions_map[drone_heading]:
        try:
            column = volume[:, int(drone_position[1]) + xy[0], int(drone_position[2]) + xy[1]]
        except IndexError:
            column = [1., 1., 1., 1., 1.]
        ego_slice[:,column_number] = column
        column_number += 1
    return np.flip(ego_slice,0)

def heading_to_hiker(drone_heading, drone_position, hiker_position):
    '''Outputs either 90Left, 45Left, 0, 45Right,90Right, or both 90Left and 90Right'''
    category_to_angle_range = {1:[0,45],2:[45,90],3:[90,135],4:[135,180],5:[180,225],6:[225,270],7:[270,315],8:[315,360]}
    category_angle = {1:0,2:45,3:90,4:135,5:180,6:225,7:270,8:315}
    drone = drone_position[-2:]
    hiker = hiker_position[-2:]
    if drone == hiker:
        return 500
    x1, x2 = drone[-2:]
    y1, y2 = hiker[-2:]

    rads = math.atan2(y1 - x1, y2 - x2)
    deg = math.degrees(rads) + 90 - (category_angle[drone_heading])
    if deg < -180:
        deg = deg + 360
    if deg > 180:
        deg = deg - 360
    return deg

def angle_categories(angle):
    '''Values -180 to +180. Returns a fuzzy set dictionary.'''
    returndict = {'hiker_left': 0, 'hiker_diagonal_left': 0, 'hiker_center': 0, 'hiker_diagonal_right': 0, 'hiker_right': 0}
    if angle < -90:
        returndict['hiker_left'] = 1
    if angle >= -90 and angle < -60:
        returndict['hiker_left'] = abs(angle + 60) / 30.0
    if angle >= -75 and angle < -45:
        returndict['hiker_diagonal_left'] = 1 + (angle + 45) / 30
    if angle >= -45 and angle < -15:
        returndict['hiker_diagonal_left'] = abs(angle + 15) / 30.0
    if angle >= - 30 and angle < 0:
        returndict['hiker_center'] = 1 + angle/30.0
    if angle >= 0 and angle < 30:
        returndict['hiker_center'] = 1 - angle/30.0
    if angle >=15 and angle < 45:
        returndict['hiker_diagonal_right'] = (angle - 15)/30.0
    if angle >=45 and angle < 75:
        returndict['hiker_diagonal_right'] = 1 - (angle - 45)/30.0
    if angle >=60 and angle < 90:
        returndict['hiker_right'] = (angle - 60)/30.0
    if angle >=90:
        returndict['hiker_right'] = 1

    if angle >= 179.9:
        returndict['hiker_right'] = 1
        returndict['hiker_left'] = 1
    if angle <= -179.9:
        returndict['hiker_right'] = 1
        returndict['hiker_left'] = 1

    if angle == 500:
        returndict['hiker_right'] = 0
        returndict['hiker_left'] = 0







    return returndict

def convert_obs_to_chunk(step):


    step['volume'] = step['map']['vol']
    egocentric_angle_to_hiker = heading_to_hiker(step['heading'], step['drone'], step['hiker'])
    angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
    egocentric_slice = egocentric_representation(step['drone'], step['heading'], step['volume'])
    # compile all that into chunks [slot, value, slot, value]
    chunk = {}
    for key, value in angle_categories_to_hiker.items():
        chunk[key] = value  # .extend([key, [key, value]])
    # need the altitudes from the slice
    altitudes = altitudes_from_egocentric_slice(egocentric_slice)
    altitudes = [x - 1 for x in altitudes]
    alt = step['altitude']
    # chunk.extend(['altitude', ['altitude', int(alt)]])
    chunk['altitude'] = int(alt)
    chunk['ego_left'] = altitudes[0]
    chunk['ego_diagonal_left'] = altitudes[1]
    chunk['ego_center'] = altitudes[2]
    chunk['ego_diagonal_right'] = altitudes[3]
    chunk['ego_right'] = altitudes[4]
    # chunk.extend(['ego_left', ['ego_left', altitudes[0]],
    #               'ego_diagonal_left', ['ego_diagonal_left', altitudes[1]],
    #               'ego_center', ['ego_center', altitudes[2]],
    #               'ego_diagonal_right', ['ego_diagonal_right', altitudes[3]],
    #               'ego_right', ['ego_right', altitudes[4]]])

    chunk['distance_to_hiker'] = distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))
    # also want distance  to hiker
    # chunk.extend(['distance_to_hiker',
    #               ['distance_to_hiker', distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))]])
    # split action into components [up, level, down, left, right, etc]
    # components = action_to_category_map[step['action']]
    # for component in components:
    #     action_values[component] = 1
    # for key, value in action_values.items():
    #     chunk.extend([key, [key, value]])

    return chunk




def convert_data_to_chunks(all_data,include_fc=False):
    nav = []
    drop = []
    for episode in all_data:
        for step in episode['nav']:
            # action_values = {'drop': 0, 'left': 0, 'diagonal_left': 0,
            #                  'center': 0, 'diagonal_right': 0, 'right': 0,
            #                  'up': 0, 'down': 0, 'level': 0}
            # angle to hiker: negative = left, positive right
            egocentric_angle_to_hiker = heading_to_hiker(step['heading'], step['drone'], step['hiker'])
            angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
            egocentric_slice = egocentric_representation(step['drone'], step['heading'], step['volume'])
            # compile all that into chunks [slot, value, slot, value]
            chunk = {}
            for key, value in angle_categories_to_hiker.items():
                chunk[key] = value#.extend([key, [key, value]])
            # need the altitudes from the slice
            altitudes = altitudes_from_egocentric_slice(egocentric_slice)
            altitudes = [x - 1 for x in altitudes]
            alt = step['altitude']
            #chunk.extend(['altitude', ['altitude', int(alt)]])
            chunk['altitude'] = int(alt)
            chunk['ego_left'] = altitudes[0]
            chunk['ego_diagonal_left'] = altitudes[1]
            chunk['ego_center'] = altitudes[2]
            chunk['ego_diagonal_right'] = altitudes[3]
            chunk['ego_right'] = altitudes[4]
            # chunk.extend(['ego_left', ['ego_left', altitudes[0]],
            #               'ego_diagonal_left', ['ego_diagonal_left', altitudes[1]],
            #               'ego_center', ['ego_center', altitudes[2]],
            #               'ego_diagonal_right', ['ego_diagonal_right', altitudes[3]],
            #               'ego_right', ['ego_right', altitudes[4]]])


            chunk['distance_to_hiker'] = distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))
            # also want distance  to hiker
            # chunk.extend(['distance_to_hiker',
            #               ['distance_to_hiker', distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))]])
            # split action into components [up, level, down, left, right, etc]
            # components = action_to_category_map[step['action']]
            # for component in components:
            #     action_values[component] = 1
            # for key, value in action_values.items():
            #     chunk.extend([key, [key, value]])

            #last part of the observation side will be the vector
            if include_fc:
                fc_list = tuple(step['fc'].tolist()[0])
                chunk['fc'] = fc_list#.extend(['fc', ['fc', fc_list]])

            # if step['action'] == 15:
            #     print('15')

            #add the action probabilities, to keep track of
            chunk['action_probs'] = step['action_probs']


            #no longer splitting the actions. Use all 15
            #actr_actions = ['_'.join(x) if len(x) > 1 else x for x in action_to_category_map.values()]
            for key,value in action_to_category_map.items():
                if len(value) > 1:
                    actr_action = '_'.join(value)
                else:
                    actr_action = value[0]
                if key == step['action']:
                    chunk[actr_action] = 1#.extend([actr_action,[actr_action,1]])
                else:
                    chunk[actr_action] = 0#.extend([actr_action,[actr_action,0]])

            # chunk.extend(['drop',['drop',drop_val]])
            # chunk.extend(['type', 'nav'])




            nav.append(chunk)


    return nav

def multi_blends(chunk, memory, slots ):
    return [memory.blend(slot,**chunk) for slot in slots]

def vector_similarity(x,y):
    return distance.euclidean(x,y) / 4.5#max(FC_distances)
    # return distance.cosine(x,y)


def custom_similarity(x,y):
    return 0
    return abs(x - y)

def check_and_handle_existing_folder(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)


def _print(i):
    print(datetime.now())
    print("# batch %d" % i)
    sys.stdout.flush()



def restore_last_move(human_data,environment):
    if len(human_data['maps']) == 0:
        return 0
    environment.reset(map=flags.FLAGS.map,config=flags.FLAGS.configuration)
    for step in human_data['actions'][:-1]:
        environment.step(step)
    human_data['maps'] = human_data['maps'][:-1]
    human_data['actions'] = human_data['actions'][:-1]
    human_data['headings'] = human_data['headings'][:-1]
    human_data['altitudes'] = human_data['altitudes'][:-1]
    human_data['drone'] = human_data['drone'][:-1]
    human_data['hiker'] = human_data['hiker'][:-1]
    #human_data['reward'] = human_data['reward'][:-1]





def main():
        score = 0.0
        reward = 0.0
        environment = GridWorld.GridworldEnv()
        config = flags.FLAGS.configuration
        environment.reset(map=flags.FLAGS.map,config=config)
        human_data = {}
        human_data['map'] = []
        human_data['actions'] = []
        human_data['headings'] = []
        human_data['altitudes'] = []
        human_data['drone'] = []
        human_data['hiker'] = []
        #human_data['reward'] = []

        episode_counter = 0

        # pygame.font.get_fonts() # Run it to get a list of all system fonts
        display_w = 1200
        display_h = 720

        BLUE = (128, 128, 255)
        DARK_BLUE = (1, 50, 130)
        RED = (255, 192, 192)
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)

        sleep_time = 0.0

        pygame.init()
        gameDisplay = pygame.display.set_mode((display_w, display_h))
        gameDisplay.fill(DARK_BLUE)
        pygame.display.set_caption('Package Drop')
        clock = pygame.time.Clock()

        def screen_mssg_variable(text, variable, area):
            font = pygame.font.SysFont('arial', 16)
            txt = font.render(text + str(variable), True, WHITE)
            gameDisplay.blit(txt, area)

        def process_img(img, x,y):
            # swap the axes else the image will not be the same as the matplotlib one
            img = img.transpose(1,0,2)
            surf = pygame.surfarray.make_surface(img)
            surf = pygame.transform.scale(surf, (300, 300))
            gameDisplay.blit(surf, (x, y))

        def text_objects(text, font):
            textSurface = font.render(text, True, (255,0,0))
            return textSurface, textSurface.get_rect()

        def message_display(text):
            largeText = pygame.font.Font('freesansbold.ttf', 115)
            TextSurf, TextRect = text_objects(text, largeText)
            TextRect.center = ((display_w / 2), (display_h / 2))
            gameDisplay.blit(TextSurf, TextRect)

            pygame.display.update()

            time.sleep(2)



        dictionary = {}
        running = True
        done = 0
        while episode_counter <= (FLAGS.episodes - 1) and running==True and done ==False:
            print('Episode: ', episode_counter)
            human_data = {}
            human_data['map'] = []
            human_data['action'] = []
            human_data['heading'] = []
            human_data['altitude'] = []
            human_data['drone'] = []
            human_data['hiker'] = []
            #human_data['reward'] = []






            new_image = environment.generate_observation()
            map_xy = new_image['img']#environment.generate_observation()['img']map_image
            map_alt = new_image['nextstepimage']#environment.alt_view
            process_img(map_xy, 20, 20)
            process_img(map_alt, 20, 400)
            pygame.display.update()


            # Quit pygame if the (X) button is pressed on the top left of the window
            # Seems that without this for event quit doesnt show anything!!!
            # Also it seems that the pygame.event.get() is responsible to REALLY updating the screen contents
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("human data being written.")
                    with open('./data/human.tj', 'wb') as handle:
                        pickle.dump(dictionary, handle)
                    running = False
            sleep(sleep_time)


            # Timestep counter
            t=0

            drop_flag = 0
            # done = 0
            while 1:#done==0:

                gameDisplay.fill(DARK_BLUE)

                #
                # drone_pos = np.where(nav_runner.envs.map_volume['vol'] == nav_runner.envs.map_volume['feature_value_map']['drone'][nav_runner.envs.altitude]['val'])

                action= -1
                #ignore mouse actions!!!
                pygame.event.set_blocked([pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    print('participant data written.')
                    timestr = timestr = time.strftime("%Y%m%d-%H%M%S")
                    with open('./data/human-data/' + flags.FLAGS.participant + '-' + timestr + '.tj', 'wb') as handle:
                        pickle.dump(human_data, handle)
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if (event.key == pygame.K_z):
                        action = 0
                    elif (event.key == pygame.K_x):
                        action = 1
                    elif (event.key == pygame.K_c):
                        action = 2
                    elif (event.key == pygame.K_v):
                        action = 3
                    elif (event.key == pygame.K_b):
                        action = 4
                    elif (event.key == pygame.K_a):
                        action = 5
                    elif (event.key == pygame.K_s):
                        action = 6
                    elif (event.key == pygame.K_d):
                        action = 7
                    elif (event.key == pygame.K_f):
                        action = 8
                    elif (event.key == pygame.K_g):
                        action = 9
                    elif (event.key == pygame.K_q):
                        action = 10
                    elif (event.key == pygame.K_w):
                        action = 11
                    elif (event.key == pygame.K_e):
                        action = 12
                    elif (event.key == pygame.K_r):
                        action = 13
                    elif (event.key == pygame.K_t):
                        action = 14
                    elif (event.key == pygame.K_SPACE):
                        action = 15
                    elif (event.key == pygame.K_DELETE):
                        restore_last_move(human_data, environment)
                        obs = environment.generate_observation()
                        score -= reward
                        map_xy = obs['img']
                        map_alt = obs['nextstepimage']
                        process_img(map_xy, 20, 20)
                        process_img(map_alt, 20, 400)
                        pygame.display.update()
                    elif (event.key == pygame.K_BACKSPACE):
                        restore_last_move(human_data,environment)
                        obs = environment.generate_observation()
                        score -= reward
                        map_xy = obs['img']
                        map_alt = obs['nextstepimage']
                        process_img(map_xy, 20, 20)
                        process_img(map_alt, 20, 400)
                        pygame.display.update()
                    else:
                        continue

                if action == -1:
                    continue
                # action stays till renewed no matter what key you press!!! So whichever key will do the last action
                pygame.event.clear()
                observation, reward, done, info = environment.step(action)

                human_data = {}
                human_data['map'] = environment.map_volume
                human_data['heading'] = environment.heading
                human_data['altitude'] = environment.altitude
                human_data['action'] = action
                drone_pos = np.where(environment.map_volume['vol'] == environment.map_volume['feature_value_map']['drone'][environment.altitude]['val'])

                human_data['drone'] = drone_pos
                human_data['hiker'] = environment.hiker_position

                chunk_observation = convert_obs_to_chunk(human_data)
                print(chunk_observation)




                #human_data['reward'].append(reward)

                score += reward

                print("DONE:", done)
                print("REWARD:", reward)

                pygame.event.set_blocked([pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
                event = pygame.event.wait() # PREVENTS FOR CONSIDERING MORE THAN A KEY PRESS AT ONCE. CAREFUL

                scoreFont = pygame.font.Font('freesansbold.ttf', 24)
                scoreText = scoreFont.render("Current Score {}".format(score), 1, (0,0,0))
                gameDisplay.blit(scoreText,(500,200))

                rewardText = scoreFont.render("Last Reward {}".format(reward), 1, (0,0,0))
                gameDisplay.blit(rewardText,(500,250))

                pygame.display.update()
                pygame.event.get()
                sleep(sleep_time)



                obs = environment.generate_observation()
                map_xy = obs['img']
                map_alt = obs['nextstepimage']
                process_img(map_xy, 20, 20)
                process_img(map_alt, 20, 400)

                # Update finally the screen with all the images you blitted in the run_trained_batch
                pygame.display.update() # Updates only the blitted parts of the screen, pygame.display.flip() updates the whole screen
                pygame.event.get() # Show the last state and then reset
                sleep(sleep_time)
                t += 1
                # if t == 70:
                #     break
                if done:
                    pygame.event.set_blocked([pygame.KEYDOWN])
                    message_display("GAME OVER")


            clock.tick(15)


        print("human data being written.")
        timestr = timestr = time.strftime("%Y%m%d-%H%M%S")
        with open('./data/human-data/' + flags.FLAGS.participant + '-' + timestr + '.tj', 'wb') as handle:
            pickle.dump(human_data, handle)




if __name__ == "__main__":
    main()
