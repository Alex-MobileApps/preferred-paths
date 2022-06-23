# cd to package top-level directory
# run: python3 -m unittest test.test_global_brain.py

from test.test import Test
from brain import Brain
import numpy as np
from numpy import inf

class TestGlobalBrain(Test):

    def test_closest_to_target(self):
        M = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
        euc = np.array([[0,3,2.8,2.2],[3,0,2.2,2.8],[2.8,2.2,0,1],[2.2,2.8,1,0]])
        brain = Brain(sc=M, fc=M, euc_dist=euc)
        exp = np.array([
            # target 0
            [[ inf,  -3, inf,-2.2],
             [   3, inf, 0.2, inf],
             [ inf,-0.2, inf, 0.6],
             [ 2.2, inf,-0.6, inf]],

            # target 1
            [[ inf,   3, inf, 0.2],
             [  -3, inf,-2.2, inf],
             [ inf, 2.2, inf,-0.6],
             [-0.2, inf, 0.6, inf]],

            # target 2
            [[ inf, 0.6, inf, 1.8],
             [-0.6, inf, 2.2, inf],
             [ inf,-2.2, inf,  -1],
             [-1.8, inf,   1, inf]],

            # target 3
            [[ inf,-0.6, inf, 2.2],
             [ 0.6, inf, 1.8, inf],
             [ inf,-1.8, inf,   1],
             [-2.2, inf,  -1, inf]]])

        test = lambda loc, nxt, target: self.assert_float(brain.closest_to_target(loc, nxt, target), exp[target,loc,nxt])
        for loc in range(4):
            for nxt in range(4):
                for target in range(4):
                    test(loc, nxt, target)
        #self.assert_float(brain.closest_to_target(), exp)