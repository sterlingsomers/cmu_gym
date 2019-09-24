
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
flags.DEFINE_string("participant", 'dfm', "The participants name")
flags.DEFINE_string("map", 'canyon', "river, canyon, v-river, treeline, small-canyon, flatland")
flags.DEFINE_integer("configuration", 2, "0,1, or 2")



# flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

#flags.DEFINE_string("map", None, "Name of a map to use.")


FLAGS(sys.argv)



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
    human_data['reward'] = human_data['reward'][:-1]



def main():
        score = 0.0
        reward = 0.0
        environment = GridWorld.GridworldEnv()
        config = flags.FLAGS.configuration
        environment.reset(map=flags.FLAGS.map,config=config)
        human_data = {}
        human_data['maps'] = []
        human_data['actions'] = []
        human_data['headings'] = []
        human_data['altitudes'] = []
        human_data['drone'] = []
        human_data['hiker'] = []
        human_data['reward'] = []

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
            human_data['maps'] = []
            human_data['actions'] = []
            human_data['headings'] = []
            human_data['altitudes'] = []
            human_data['drone'] = []
            human_data['hiker'] = []
            human_data['reward'] = []






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

                human_data['maps'].append(environment.map_volume)
                human_data['headings'].append(environment.heading)
                human_data['altitudes'].append(environment.altitude)
                human_data['actions'].append(action)
                drone_pos = np.where(environment.map_volume['vol'] == environment.map_volume['feature_value_map']['drone'][environment.altitude]['val'])

                human_data['drone'].append(drone_pos)
                human_data['hiker'].append(environment.hiker_position)




                observation, reward, done, info = environment.step(action)

                human_data['reward'].append(reward)

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
