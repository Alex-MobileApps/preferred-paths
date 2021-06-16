# cd to package top-level directory
# run: python3 -m unittest test.test_brain.py

from brain import Brain
import numpy as np
from utils import binarise_matrix
from test.test import Test

class TestBrain(Test):

    @classmethod
    def setUp(cls):
        load = lambda txt: np.loadtxt(f'test/test_{txt}.txt', delimiter=',')
        cls.sc1 = load('sc1')
        cls.sc2 = load('sc2')
        cls.fc1 = load('fc1')
        cls.fc2 = load('fc2')
        cls.euc_dist = load('euc_dist')
        cls.sc_thresh1 = 1
        cls.sc_thresh2 = 5
        cls.fc_thresh1 = 0.01
        cls.fc_thresh2 = 0.1
        cls.sc_thresh_type1 = 'pos'
        cls.sc_thresh_type2 = 'neg'
        cls.fc_thresh_type1 = 'pos'
        cls.fc_thresh_type2 = 'abs'
        cls.res = 20
        cls.brain1 = Brain(cls.sc1, cls.fc1, cls.euc_dist, cls.sc_thresh1, cls.fc_thresh1)
        cls.brain2 = Brain(cls.sc2, cls.fc2, cls.euc_dist, cls.sc_thresh2, cls.fc_thresh2, cls.sc_thresh_type2, cls.fc_thresh_type2)

    def test_init(self):
        # Failed brains
        non_square = np.array([[0,1]])
        directed = np.array([[0,1],[2,0]])
        non_np = [[0,1],[1,0]]
        small = np.array(non_np)
        self.assert_raise(True, lambda: Brain(self.sc1, non_np, self.euc_dist, 1, 1))     # Non numpy
        self.assert_raise(True, lambda: Brain(non_np, self.fc1, self.euc_dist, 1, 1))
        self.assert_raise(True, lambda: Brain(self.sc1, self.fc1, non_np, 1, 1))
        self.assert_raise(True, lambda: Brain(self.sc1, non_square, self.euc_dist, 1, 1)) # Non square
        self.assert_raise(True, lambda: Brain(non_square, self.fc1, self.euc_dist, 1, 1))
        self.assert_raise(True, lambda: Brain(self.sc1, self.fc1, non_square, 1, 1))
        self.assert_raise(True, lambda: Brain(self.sc1, small, self.euc_dist, 1, 1))      # Resolution mismatch
        self.assert_raise(True, lambda: Brain(small, self.fc1, self.euc_dist, 1, 1))
        self.assert_raise(True, lambda: Brain(self.sc1, self.fc1, small, 1, 1))

        loopless = lambda M: M * (1 - np.eye(self.res))
        test_wei = lambda a, b: self.assert_float(a, loopless(b))
        test_bin = lambda a, b, c, d: self.assert_float(a, loopless(binarise_matrix(b, c, d)))

        # Brain 1
        np.testing.assert_equal(self.brain1.sc_thresh, self.sc_thresh1)
        np.testing.assert_equal(self.brain1.fc_thresh, self.fc_thresh1)
        np.testing.assert_equal(self.brain1.sc_thresh_type, self.sc_thresh_type1)
        np.testing.assert_equal(self.brain1.fc_thresh_type, self.fc_thresh_type1)
        np.testing.assert_equal(self.brain1.res, self.res)
        test_wei(self.brain1.sc, self.sc1)
        test_wei(self.brain1.fc, self.fc1)
        test_wei(self.brain1.euc_dist, self.euc_dist)
        test_bin(self.brain1.sc_bin, self.sc1, self.sc_thresh1, self.sc_thresh_type1)
        test_bin(self.brain1.fc_bin, self.fc1, self.fc_thresh1, self.fc_thresh_type1)

        # Brain 2
        np.testing.assert_equal(self.brain2.sc_thresh, self.sc_thresh2)
        np.testing.assert_equal(self.brain2.fc_thresh, self.fc_thresh2)
        np.testing.assert_equal(self.brain2.sc_thresh_type, self.sc_thresh_type2)
        np.testing.assert_equal(self.brain2.fc_thresh_type, self.fc_thresh_type2)
        np.testing.assert_equal(self.brain2.res, self.res)
        test_wei(self.brain2.sc, self.sc2)
        test_wei(self.brain2.fc, self.fc2)
        test_wei(self.brain2.euc_dist, self.euc_dist)
        test_bin(self.brain2.sc_bin, self.sc2, self.sc_thresh2, self.sc_thresh_type2)
        test_bin(self.brain2.fc_bin, self.fc2, self.fc_thresh2, self.fc_thresh_type2)

    def test_streamlines(self):
        test = lambda brain, res, weighted=True: self.assert_float(brain.streamlines(weighted), res)
        test(self.brain1, self.sc1)
        test(self.brain2, self.sc2)
        test(self.brain1, self.sc1, True)
        test(self.brain2, self.sc2, True)
        test(self.brain1, (self.sc1 >= 1).astype(np.float), False)

    def test_edge_length(self):
        test = lambda brain, res: self.assert_float(brain.edge_length(), res)
        test(self.brain1, self.euc_dist)
        test(self.brain2, self.euc_dist)

    def test_edge_angle_change(self):
        test = lambda brain, source, target, prev, res: self.assert_float(brain.edge_angle_change(source, target, prev), res)
        test(self.brain1, 0, 1, [], 0)       # No previous
        test(self.brain1, 0, 0, [1], 0)      # Same source/destination
        test(self.brain1, 0, 1, [2], 1.91)
        test(self.brain1, 0, 1, [3,2], 1.91) # Longer path
        test(self.brain1, 6, 7, [8], 2.58)   # New value

    def test_node_strength(self):
        test = lambda brain, res, weighted=True: self.assert_float(brain.node_strength(weighted)[3], res)
        test(self.brain1, 2336.94)
        test(self.brain1, 2336.94, True)
        test(self.brain2, 2489.74)
        test(self.brain2, 2489.74, True)
        test(self.brain1, 15, False)

    def test_node_strength_disimilarity(self):
        test = lambda brain, res, weighted=True: self.assert_float(brain.node_strength_disimilarity(weighted)[3,5], res)
        test(self.brain1, 4431.02)
        test(self.brain1, 4431.02, True)
        test(self.brain2, 1864.87)
        test(self.brain2, 1864.87, True)
        test(self.brain1, 2, False)

    def test_triangle_node_prevalence(self):
        test = lambda brain, res: self.assert_float(brain.triangle_node_prevalence()[5], res)
        test(self.brain1, 2)
        test(self.brain2, 4)

    def test_triangle_edge_prevalence(self):
        test = lambda brain, source, target, res: self.assert_float(brain.triangle_edge_prevalence()[source, target], res)
        test(self.brain1, 2, 18, 2) # 2-0-18, 2-7-18
        test(self.brain1, 0, 5, 1)  # 0-18-5
        test(self.brain1, 0, 3, 0)
        test(self.brain1, 1, 1, 0) # Same node

    def test_dist_to_prev_used(self):
        test = lambda brain, prev, res: self.assert_float(brain.dist_to_prev_used(5, prev), res)
        test(self.brain1,    [], 0)        # No fail on empty
        test(self.brain1, [0,2], 3855)     # Choose closest
        test(self.brain1, [0,2,6], 3855)   # Added further apart node
        test(self.brain1, [0,2,6,3], 3761) # Added closer node
        test(self.brain1, [0,3,2,6], 3761) # Change order

if __name__ == '__main__':
    TestBrain.main()