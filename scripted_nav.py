import pygame
import time
import os
from time import sleep
import numpy as np
import gym
import gym_gridworld
import random

def get_nextstep_altitudes(map_volume,drone_position,heading):
   slice = np.zeros((5, 5))
   drone_position_flat = [int(drone_position[1]), int(drone_position[2])]

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
   column_number = 0
   for xy in possible_actions_map[heading]:
       try:
           # no hiker if using original
           column = map_volume['vol'][:, drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]
           # for p in column:
           #     #print(p)
           #     #print(p == 50.0)
           #     if p == 50.0: # Hiker representation in the volume
           #         #print("setting hiker_found to True")
           #         hiker_found = True
           #
           # if hiker_found:
           #     val = self.original_map_volume['vol'][0][
           #         drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]
           #     hiker_background_color = self.original_map_volume['value_feature_map'][val]['color']
           #     # column = self.original_map_volume['vol'][:,drone_position_flat[0]+xy[0],drone_position_flat[1]+xy[1]]
       except IndexError:
           column = [1., 1., 1., 1., 1.]
       slice[:, column_number] = column
       column_number += 1
       # print("ok")
   # put the drone in
   # cheat
   corrected_slice = np.flip(slice,0)
   #one = np.count_nonzero(corrected_slice)
   two = np.count_nonzero(corrected_slice, axis=0)
   #three = np.count_nonzero(corrected_slice, axis=1)
   return two

def main():
    EPISODES = 5
    ACTS = [5, 6, 7, 8, 9]
    envs = gym.make('gridworld-v0')

    print("Requested environments created successfully")

    display_w = 800
    display_h = 720

    BLUE = (128, 128, 255)
    DARK_BLUE = (1, 50, 130)
    RED = (255, 192, 192)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    pygame.init()
    gameDisplay = pygame.display.set_mode((display_w, display_h))
    gameDisplay.fill(DARK_BLUE)
    pygame.display.set_caption('Neural Introspection')
    clock = pygame.time.Clock()

    def screen_mssg_variable(text, variable, area):
        font = pygame.font.SysFont('arial', 16)
        txt = font.render(text + str(variable), True, WHITE)
        gameDisplay.blit(txt, area)
        # pygame.display.update()

    def process_img(img, x, y):
        # swap the axes else the image will come not the same as the matplotlib one
        img = img.transpose(1, 0, 2)
        surf = pygame.surfarray.make_surface(img)
        surf = pygame.transform.scale(surf, (300, 300))
        gameDisplay.blit(surf, (x, y))

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
    epis = 1
    while epis <= EPISODES:
        print('Episode: ', epis)
        obs = envs.reset()  # Cauz of differences in the arrangement of the dictionaries
        map_xy = obs['img']
        map_alt = obs['nextstepimage']
        process_img(map_xy, 20, 20)
        process_img(map_alt, 20, 400)
        pygame.display.update()
        # Quit pygame if the (X) button is pressed on the top left of the window
        # Seems that without this for event quit doesnt show anything!!!
        # Also it seems that the pygame.event.get() is responsible to REALLY updating the screen contents
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        sleep(1.5)
        # Timestep counter
        t = 0
        rewards = []
        done = 0
        while done == 0:
            # RUN THE MAIN LOOP

            ####### START CODING MESS #######
            #action = agent.step()
            hiker_pos = envs.hiker_position[1:] # Take out altitude
            heading = envs.heading
            altitude = envs.altitude
            map_volume = obs['volume']
            drone_pos = np.where(
                map_volume['vol'] == map_volume['feature_value_map']['drone'][altitude]['val'])

            alts = get_nextstep_altitudes(map_volume, drone_pos, heading)
            mask = [0, 0, 0, 0, 0]
            i = 0
            for x in alts:
                if x <= 3:
                    mask[i] = 1
                else:
                    mask[i] = 0
                i = i + 1
            # multiply with mask to get available actions
            avail_acts = [a * b for a, b in zip(ACTS, mask)]
            # Get available actions indices
            indx = [i for i, e in enumerate(avail_acts) if e != 0]
            # Get displacements
            dxdy = possible_actions_map[heading]
            distance = 1000
            len_indx = len(indx)
            for j in range(0,len_indx):
                next_st = [drone_pos[1:][0][0] + dxdy[j][0] , drone_pos[1:][1][0] + dxdy[j][1]]
                dist = max([ abs(next_st[0]-hiker_pos[0][0]) , abs(next_st[1]-hiker_pos[1][0])])
                if (dist < distance):
                    # print('j',j, 'indx', indx)
                    # print("indx of action to be taken", indx[j])
                    min_dist_ind = indx[j]
                    distance = dist
            ####### END OF CODING MESS ########

            action = ACTS[min_dist_ind]
            # action = random.randint(0,14)
            obs, reward, done, info = envs.step(action)

            rewards.append(reward)
            if done:
                score = sum(rewards)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>> episode %d ended in %d steps. Score %f" % (
                    epis, t, score))


            # screen_mssg_variable("Value    : ", np.round(value, 3), (168, 350))
            # screen_mssg_variable("Reward: ", np.round(reward, 3), (168, 372))
            # pygame.display.update()
            # pygame.event.get()
            # sleep(1.5)

            # BLIT!!!
            # First Background covering everyything from previous session
            gameDisplay.fill(DARK_BLUE)
            map_xy = obs['img']
            map_alt = obs['nextstepimage']
            process_img(map_xy, 20, 20)
            process_img(map_alt, 20, 400)
            # Update finally the screen with all the images you blitted in the run_trained_batch
            pygame.display.update()  # Updates only the blitted parts of the screen, pygame.display.flip() updates the whole screen
            pygame.event.get()  # Show the last state and then reset
            sleep(1.2)
            t += 1

        epis +=1
        clock.tick(15)

    print("Okay. Work is done")
    envs.close()

if __name__ == "__main__":
    main()
