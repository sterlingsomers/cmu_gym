import unittest
from gym_gridworld.envs.gridworld_env import ACTION, HEADING

class TestActionEnumeration(unittest.TestCase):


    def test_action_forward(self):

        new_heading = ACTION.new_heading( HEADING.NORTH, ACTION.LEVEL_FORWARD )
        self.assertEqual( new_heading, HEADING.NORTH )

        new_heading = ACTION.new_heading( HEADING.SOUTH, ACTION.LEVEL_FORWARD )
        self.assertEqual( new_heading, HEADING.SOUTH )

        new_heading = ACTION.new_heading( HEADING.EAST, ACTION.LEVEL_FORWARD )
        self.assertEqual( new_heading, HEADING.EAST )

        new_heading = ACTION.new_heading( HEADING.WEST, ACTION.LEVEL_FORWARD )
        self.assertEqual( new_heading, HEADING.WEST )


    def test_action_right_45(self):

        new_heading = ACTION.new_heading( HEADING.NORTH, ACTION.LEVEL_RIGHT_45 )
        self.assertEqual( new_heading, HEADING.NORTH_EAST )

        new_heading = ACTION.new_heading( HEADING.NORTH_EAST, ACTION.LEVEL_RIGHT_45 )
        self.assertEqual( new_heading, HEADING.EAST )

        new_heading = ACTION.new_heading( HEADING.SOUTH, ACTION.LEVEL_RIGHT_45 )
        self.assertEqual( new_heading, HEADING.SOUTH_WEST )

        new_heading = ACTION.new_heading( HEADING.NORTH_WEST, ACTION.LEVEL_RIGHT_45 )
        self.assertEqual( new_heading, HEADING.NORTH )


    def test_action_right_90(self):

        new_heading = ACTION.new_heading( HEADING.NORTH, ACTION.LEVEL_RIGHT_90 )
        self.assertEqual( new_heading, HEADING.EAST )

        new_heading = ACTION.new_heading( HEADING.NORTH_EAST, ACTION.LEVEL_RIGHT_90 )
        self.assertEqual( new_heading, HEADING.SOUTH_EAST )

        new_heading = ACTION.new_heading( HEADING.SOUTH, ACTION.LEVEL_RIGHT_90 )
        self.assertEqual( new_heading, HEADING.WEST )

        new_heading = ACTION.new_heading( HEADING.NORTH_WEST, ACTION.LEVEL_RIGHT_90 )
        self.assertEqual( new_heading, HEADING.NORTH_EAST )


if __name__ == '__main__':
    unittest.main()

