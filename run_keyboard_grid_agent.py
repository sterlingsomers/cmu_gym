import os
import shutil
import sys
from datetime import datetime
from time import sleep
import pickle
import pygame, time, random
import math
from absl import flags
from gridworld_v2 import gameEnv
from scipy.spatial import distance
# import gym_gridworld.envs.gridworld_env as GridWorld

FLAGS = flags.FLAGS
# flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
# flags.DEFINE_bool("Save", False, "Whether to save the collected data of the agents.")
flags.DEFINE_integer("resolution", 400, "Resolution for task image.")
# flags.DEFINE_integer("step_mul", 100, "Game steps per agent step.")
# flags.DEFINE_integer("n_envs", 20, "Number of environments to run in parallel")
flags.DEFINE_integer("episodes", 2, "Number of complete episodes")
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

flags.DEFINE_string("participant", 'Test', "The participants name")

# flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

# flags.DEFINE_string("map", None, "Name of a map to use.")


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


def main():
    env = gameEnv(partial=False, size=9)
    episode_counter = 0

    # pygame.font.get_fonts() # Run it to get a list of all system fonts
    display_w = 450#1200
    display_h = 450#720
    resolution = FLAGS.resolution

    BLUE = (128, 128, 255)
    DARK_BLUE = (1, 50, 130)
    RED = (255, 192, 192)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    sleep_time = 0.0

    pygame.init()
    gameDisplay = pygame.display.set_mode((display_w, display_h))
    gameDisplay.fill(DARK_BLUE)
    pygame.display.set_caption('Neural Introspection')
    clock = pygame.time.Clock()

    def screen_mssg_variable(text, variable, area):
        font = pygame.font.SysFont('arial', 16)
        txt = font.render(text + str(variable), True, WHITE)
        gameDisplay.blit(txt, area)

    def process_img(img, x, y, dims):
        # swap the axes else the image will not be the same as the matplotlib one
        img = img.transpose(1, 0, 2)
        surf = pygame.surfarray.make_surface(img)
        surf = pygame.transform.scale(surf, (dims, dims))
        gameDisplay.blit(surf, (x, y))

    def text_objects(text, font):
        textSurface = font.render(text, True, (0, 0, 0))
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
    step_data = []
    while episode_counter <= (FLAGS.episodes - 1):# and running == True: #and done == False:
        print('Episode: ', episode_counter)
        # human_data = {}
        # human_data['maps'] = []
        # human_data['actions'] = []
        # human_data['headings'] = []
        # human_data['altitudes'] = []

        # Init storage structures
        # dictionary[nav_runner.episode_counter] = {}
        # mb_obs = []
        # mb_actions = []
        # mb_flag = []
        # mb_representation = []
        # mb_fc = []
        # mb_rewards = []
        # mb_values = []
        # mb_drone_pos = []
        # mb_heading = []
        # mb_crash = []
        # mb_map_volume = [] # obs[0]['volume']==envs.map_volume
        # mb_ego = []
        env.reset()
        obs = env.renderEnv()
        map_xy = obs['img']  # env.generate_observation()['img']map_image
        # map_alt = new_image['nextstepimage']  # env.alt_view
        process_img(map_xy, 20, 20, resolution)
        # process_img(map_alt, 20, 400)
        pygame.display.update()

        # dictionary[nav_runner.episode_counter]['hiker_pos'] = nav_runner.envs.hiker_position
        # dictionary[nav_runner.episode_counter]['map_volume'] = map_xy

        # Quit pygame if the (X) button is pressed on the top left of the window
        # Seems that without this for event quit doesnt show anything!!!
        # Also it seems that the pygame.event.get() is responsible to REALLY updating the screen contents
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("human data being written.")
                # with open('./data/human.tj', 'wb') as handle:
                #     pickle.dump(dictionary, handle)
                running = False
        sleep(sleep_time)

        # Timestep counter
        t = 0

        drop_flag = 0
        done = 0

        while done==0:

            # mb_obs.append(nav_runner.latest_obs)
            # mb_flag.append(drop_flag)
            # mb_heading.append(nav_runner.envs.heading)
            #
            # drone_pos = np.where(nav_runner.envs.map_volume['vol'] == nav_runner.envs.map_volume['feature_value_map']['drone'][nav_runner.envs.altitude]['val'])
            # mb_drone_pos.append(drone_pos)
            # mb_map_volume.append(nav_runner.envs.map_volume)
            # mb_ego.append(nav_runner.envs.ego)
            action = -1
            # ignore mouse actions!!!
            pygame.event.set_blocked([pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                print('participant data written.')
                timestr = timestr = time.strftime("%Y%m%d-%H%M%S")
                with open('./data/human-data/fire-world/' + flags.FLAGS.participant + '-' + timestr + '.tj', 'wb') as handle:
                    pickle.dump(step_data, handle)
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_UP):
                    action = 0
                elif (event.key == pygame.K_DOWN):
                    action = 1
                elif (event.key == pygame.K_LEFT):
                    action = 2
                elif (event.key == pygame.K_RIGHT):
                    action = 3
                else:
                    continue

            if action == -1:
                continue
            # action stays till renewed no matter what key you press!!! So whichever key will do the last action
            pygame.event.clear()

            # human_data['maps'].append(env.map_volume)
            # human_data['headings'].append(env.heading)
            # human_data['altitudes'].append(env.altitude)
            # human_data['actions'].append(action)
            print('')
            step = {}

            for thing in env.objects:
                if thing.name == 'hero':
                    step['hero'] = (thing.x,thing.y)
                    print('')
                if thing.name == 'goal':
                    step['goal'] = (thing.x,thing.y)
                if thing.name == 'fire':
                    fire_val = 0
                    for key in step.keys():
                        if 'fire' in key:
                            fire_val += 1
                    fire_str = 'fire' + repr(fire_val)
                    step[fire_str] = (thing.x,thing.y)

            # r = distance.euclidean(hero,goal)
            # #angle = math.acos((hero[0] - goal[0])/r)
            # #angle = math.atan2(hero[0]-goal[0], hero[1]-goal[1])
            # angle = math.atan2(goal[1]-hero[1], goal[0]-hero[0])

            step['action'] = action
            step_data.append(step)
            print('debug')






            observation, reward, done, info = env.step(action)

            print("DONE:", done)

            pygame.event.set_blocked([pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
            event = pygame.event.wait()  # PREVENTS FOR CONSIDERING MORE THAN A KEY PRESS AT ONCE. CAREFUL
            # screen_mssg_variable("Value    : ", np.round(value,3), (168, 350))
            # screen_mssg_variable("Reward: ", np.round(reward,3), (168, 372))
            pygame.display.update()
            pygame.event.get()
            sleep(sleep_time)

            # BLIT!!!
            # First Background covering everything from previous session
            gameDisplay.fill(DARK_BLUE)

            obs = env.renderEnv()
            map_xy = obs['img']
            # map_alt = obs['nextstepimage']
            process_img(map_xy, 20, 20, resolution)
            # process_img(map_alt, 20, 400)

            # Update finally the screen with all the images you blitted in the run_trained_batch
            pygame.display.update()  # Updates only the blitted parts of the screen, pygame.display.flip() updates the whole screen
            pygame.event.get()  # Show the last state and then reset
            sleep(sleep_time)
            t += 1
            # if t == 70:
            #     break
        if episode_counter <= (FLAGS.episodes - 1):#done:
            pygame.event.set_blocked([pygame.KEYDOWN])
                # message_display("GAME OVER")

        clock.tick(15)

    print("human data being written.")
    timestr = timestr = time.strftime("%Y%m%d-%H%M%S")
    # with open('./data/human-data/' + flags.FLAGS.participant + '-' + timestr + '.tj', 'wb') as handle:
    #      pickle.dump(step_data, handle)


if __name__ == "__main__":
    main()
