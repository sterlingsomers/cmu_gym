import gym_gridworld.envs.mavsim_lib_server as mavlib
import unittest
import os
import matplotlib.pyplot as plt

class MavsimLibHandlerTests(unittest.TestCase):

    def test_creation(self):

        os.chdir("..")

        path = os.getcwd()
        print("Test executing in {}".format(path))

        mavsim = mavlib.MavsimLibHandler('kingdom-base-A-7')

        mavsim.reset()

        mavsim.map_show()

        map = mavsim.get_submap( (0,0),(20,20) )

        numpy = map['img']

        plt.imshow(numpy)
        plt.show()


