# cd to package top-level directory
# run: python3 -m unittest test.test_global_brain.py

from test.test import Test
from brain import GlobalBrain

class TestGlobalBrain(Test):

    def test_closest_to_target(self):
        raise NotImplementedError("Global tests not yet created")