# cd to package top-level directory
# run: python3 -m unittest test.test_brain.py

from brain import Brain
import numpy as np
from utils import binarise_matrix
from test.test import Test
from math import pi
from numpy import inf

class TestBrain(Test):

    def test_init(self):
        # Sample Graphs
        sc = np.array([[9,-3,4],[-2,-9,2],[1,0,0]])
        fc = -sc
        euc_dist = 2 * sc

        # Failed brains
        non_square = np.array([[0,1]])
        non_np = [[0,1],[1,0]]
        small = np.array(non_np)
        self.assert_raise(True, lambda: Brain(sc,         non_np,     euc_dist, 1, 1))      # Non numpy
        self.assert_raise(True, lambda: Brain(non_np,     fc,         euc_dist, 1, 1))
        self.assert_raise(True, lambda: Brain(sc,         fc,         non_np, 1, 1))
        self.assert_raise(True, lambda: Brain(sc,         non_square, euc_dist, 1, 1))      # Non square
        self.assert_raise(True, lambda: Brain(non_square, fc,         euc_dist, 1, 1))
        self.assert_raise(True, lambda: Brain(sc,         fc,         non_square, 1, 1))
        self.assert_raise(True, lambda: Brain(sc,         small,      euc_dist, 1, 1))      # Resolution mismatch
        self.assert_raise(True, lambda: Brain(small,      fc,         euc_dist, 1, 1))
        self.assert_raise(True, lambda: Brain(sc,         fc,         small, 1, 1))

        # Brain
        loopless = lambda M: M * (1 - np.eye(len(M)))
        test_wei = lambda a, b: self.assert_float(a, loopless(b))
        test_bin = lambda a, b, c, d: self.assert_float(a, loopless(binarise_matrix(b, c, d)))
        brain = Brain(sc=sc, fc=fc, euc_dist=euc_dist, sc_directed=False)
        np.testing.assert_equal(brain.sc_directed, False)
        brain = Brain(sc=sc, fc=fc, euc_dist=euc_dist, sc_directed=True, sc_thresh=2, fc_thresh=-2, sc_thresh_type='pos', fc_thresh_type='neg')
        test_wei(brain.sc, sc)
        test_wei(brain.fc, fc)
        test_wei(brain.euc_dist, euc_dist)
        test_bin(brain.sc_bin, sc,  2, 'pos')
        test_bin(brain.fc_bin, fc, -2, 'neg')
        np.testing.assert_equal(brain.sc_directed, True)
        np.testing.assert_equal(brain.sc_thresh, 2)
        np.testing.assert_equal(brain.fc_thresh, -2)
        np.testing.assert_equal(brain.sc_thresh_type, 'pos')
        np.testing.assert_equal(brain.fc_thresh_type, 'neg')
        np.testing.assert_equal(brain.res, 3)

    def test_streamlines(self):
        M = np.array([[0,5],[0.5,0]])
        brain = Brain(sc=M, fc=M, euc_dist=M)
        test = lambda brain, weighted, exp: self.assert_float(brain.streamlines(weighted), exp)
        test(brain, True, M)
        test(brain, False, M >= 1)

    def test_edge_length(self):
        M = np.array([[0,3,5],[3,0,4],[5,4,0]])
        brain = Brain(sc=M, fc=M, euc_dist=M)
        self.assert_float(brain.edge_length(), M)

    def test_edge_angle_change(self):
        M = np.array([[0,1,2],[1,0,np.sqrt(3)],[2,np.sqrt(3),0]])
        brain = Brain(sc=M, fc=M, euc_dist=M)
        test = lambda brain, source, target, prev, exp: self.assert_float(brain.edge_angle_change(source, target, prev), exp)
        test(brain, 0, 0, [], 0)            # No previous
        test(brain, 1, 1, [0], 0)           # Same source/target
        test(brain, 0, 1, [2], 2*pi/3)
        test(brain, 1, 2, [0], pi/2)
        test(brain, 1, 2, [2], pi)          # U-turn
        test(brain, 0, 1, [1,2], 2*pi/3)    # Multi prev

    def test_node_strength(self):
        M = np.array([[0,4,3,0],[5,0,0.5,0],[0,0,0,0],[0,0,0,0]])
        brain_dir = Brain(sc=M, fc=M, euc_dist=M, sc_directed=True)
        test = lambda brain, weighted, method, exp: self.assert_float(brain.node_strength(weighted, method), exp)
        test(brain_dir,  True, 'tot', np.array([12,9.5,3.5,0]))
        test(brain_dir,  True,  'in', np.array([5,4,3.5,0]))
        test(brain_dir,  True, 'out', np.array([7,5.5,0,0]))
        test(brain_dir, False, 'tot', np.array([3,2,1,0]))
        test(brain_dir, False,  'in', np.array([1,1,1,0]))
        test(brain_dir, False, 'out', np.array([2,1,0,0]))
        M = np.array([[0,5,4,0],[5,0,0,0.5],[4,0,0,0],[0,0.5,0,0]])
        brain_und = Brain(sc=M, fc=M, euc_dist=M, sc_directed=False)
        for method in ['tot','in','out']:
            test(brain_und,  True, method, np.array([9,5.5,4,0.5]))
            test(brain_und, False, method, np.array([2,1,1,0]))

    def test_node_strength_dissimilarity(self):
        M = np.array([[0,4,3,0],[5,0,0.5,0],[0,0,0,0],[0,0,0,0]])
        brain_dir = Brain(sc=M, fc=M, euc_dist=M, sc_directed=True)
        test = lambda brain, weighted, method, exp: self.assert_float(brain.node_strength_dissimilarity(weighted, method), exp)
        test(brain_dir,  True, 'tot', np.array([[0,2.5,8.5,12],[2.5,0,6,9.5],[8.5,6,0,3.5],[12,9.5,3.5,0]]))
        test(brain_dir,  True,  'in', np.array([[0,1,1.5,5],[1,0,0.5,4],[1.5,0.5,0,3.5],[5,4,3.5,0]]))
        test(brain_dir,  True, 'out', np.array([[0,1.5,7,7],[1.5,0,5.5,5.5],[7,5.5,0,0],[7,5.5,0,0]]))
        test(brain_dir,  False, 'tot', np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]]))
        test(brain_dir,  False, 'in', np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,1,1,0]]))
        test(brain_dir,  False, 'out', np.array([[0,1,2,2],[1,0,1,1],[2,1,0,0],[2,1,0,0]]))
        M = np.array([[0,5,4,0],[5,0,0,0.5],[4,0,0,0],[0,0.5,0,0]])
        brain_und = Brain(sc=M, fc=M, euc_dist=M, sc_directed=False)
        for method in ['tot','in','out']:
            test(brain_und,  True, method, np.array([[0,3.5,5,8.5],[3.5,0,1.5,5],[5,1.5,0,3.5],[8.5,5,3.5,0]]))
            test(brain_und, False, method, np.array([[0,1,1,2],[1,0,0,1],[1,0,0,1],[2,1,1,0]]))

    def test_triangle_node_prevalence(self):
        sc = np.array([[0,0,3,0,0],[0,0,0,0,0],[4,5,0,2,0],[0,6,7,0,8],[0.5,0,0,1,0]])
        fc = np.array([[0,1,0,3,0],[1,0,0,4,0.5],[0,0,0,0,5],[3,4,0,0,0],[0,0.5,5,0,0]])
        brain = Brain(sc=sc, fc=fc, euc_dist=sc, sc_directed=True, fc_thresh=1)
        self.assert_float(brain.triangle_node_prevalence(), np.array([0,0,3,2,0]))
        sc = np.array([[0,1,2,3,4],[1,0,0.5,0,5],[2,0.5,0,0,0],[3,0,0,0,6],[4,5,0,6,0]])
        fc = np.array([[0,0,0,1,0],[0,0,0,2,0],[0,0,0,0.5,3],[1,2,0.5,0,4],[0,0,3,4,0]])
        brain = Brain(sc=sc, fc=fc, euc_dist=sc, sc_directed=False, fc_thresh=1)
        self.assert_float(brain.triangle_node_prevalence(), np.array([2,0,0,0,1]))

    def test_triangle_edge_prevalence(self):
        sc = np.array([[0,0,3,0,0],[0,0,0,0,0],[4,5,0,2,0],[0,6,7,0,8],[0.5,0,0,1,0]])
        fc = np.array([[0,1,0,3,0],[1,0,0,4,0.5],[0,0,0,0,5],[3,4,0,0,0],[0,0.5,5,0,0]])
        brain = Brain(sc=sc, fc=fc, euc_dist=sc, sc_directed=True, fc_thresh=1)
        self.assert_float(brain.triangle_edge_prevalence(), np.array([[0,0,2,0,0],[0,0,0,0,0],[1,1,0,2,0],[0,0,2,0,1],[0,0,0,1,0]]))
        sc = np.array([[0,1,2,3,4],[1,0,0.5,0,5],[2,0.5,0,0,0],[3,0,0,0,6],[4,5,0,6,0]])
        fc = np.array([[0,0,0,1,0],[0,0,0,2,0],[0,0,0,0.5,3],[1,2,0.5,0,4],[0,0,3,4,0]])
        brain = Brain(sc=sc, fc=fc, euc_dist=sc, sc_directed=False, fc_thresh=1)
        self.assert_float(brain.triangle_edge_prevalence(), np.array([[0,1,1,1,1],[1,0,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,0,1,0]]))

    def test_hops_to_prev_used(self):
        M = np.array([[0,1,0,0,0],[2,0,3,0,0],[0,0,0,4,0],[0,6,0,0,0],[0,0,0,0,0]])
        brain = Brain(sc=M, fc=M, euc_dist=M, sc_directed=True)
        test = lambda brain, target, prev, exp: self.assert_float(brain.hops_to_prev_used(target, prev), exp)
        test(brain, 0, [], 0)       # No prev
        test(brain, 0, [0], 0)      # Self
        test(brain, 4, [0], inf)    # Disconnected
        test(brain, 0, [2], 3)
        test(brain, 0, [2,3], 2)
        test(brain, 0, [3,2], 2)
        test(brain, 3, [0], 3)
        test(brain, 3, [0,2], 1)
        M = np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,0]])
        brain = Brain(sc=M, fc=M, euc_dist=M, sc_directed=False)
        test(brain, 0, [], 0)       # No prev
        test(brain, 0, [0], 0)      # Self
        test(brain, 4, [0], inf)    # Disconnected
        test(brain, 0, [2,3], 2)
        test(brain, 2, [0,3], 1)

    def test_dist_to_prev_used(self):
        M = np.array([[0,4,3,5],[4,0,5,3],[3,5,0,4],[5,3,4,0]])
        brain = Brain(sc=M, fc=M, euc_dist=M)
        test = lambda brain, target, prev, exp: self.assert_float(brain.dist_to_prev_used(target, prev), exp)
        test(brain, 0, [], 0)       # No fail on empty
        test(brain, 0, [0,1], 0)    # Target in prev
        test(brain, 1, [0,3], 3)    # Choose closest
        test(brain, 1, [3,0], 3)    # Change order
        test(brain, 2, [1,3], 4)
        test(brain, 2, [0,1,3], 3)
        test(brain, 1, [0,2,3], 3)

    def test_is_target(self):
        M = np.array([[0,1,2],[3,0,0],[0,0,0]])
        brain = Brain(sc=M, fc=M, euc_dist=M)
        test = lambda brain, nxt, target, exp: self.assertEqual(brain.is_target(nxt, target), exp)
        test(brain, 0, 0, True)
        test(brain, 0, 2, False)
        test(brain, 2, 0, False)
        test(brain, 1, 1, True)
        test(brain, 0, 1, False)
        test(brain, 1, 0, False)

    def test_shortest_paths(self):
        M = np.array([[0,2,0,0,0],[0,0,8,3,0],[1,9,0,4,0],[0,0,4,0,0],[0,0,0,0,0]])
        brain = Brain(sc=M, fc=M, euc_dist=M*2, sc_directed=True)
        self.assert_float(brain.shortest_paths('hops'), np.array([[0,1,2,2,inf],[2,0,1,1,inf],[1,1,0,1,inf],[2,2,1,0,inf],[inf,inf,inf,inf,0]]))
        self.assert_float(brain.shortest_paths('dist'), np.array([[0,4,18,10,inf],[16,0,14,6,inf],[2,6,0,8,inf],[10,14,8,0,inf],[inf,inf,inf,inf,0]]))
        M = np.array([[0,2,1,0,0],[2,0,9,0,0],[1,9,0,4,0],[0,0,4,0,0],[0,0,0,0,0]])
        brain = Brain(sc=M, fc=M, euc_dist=M*2, sc_directed=False)
        self.assert_float(brain.shortest_paths('hops'), np.array([[0,1,1,2,inf],[1,0,1,2,inf],[1,1,0,1,inf],[2,2,1,0,inf],[inf,inf,inf,inf,0]]))
        self.assert_float(brain.shortest_paths('dist'), np.array([[0,4,2,10,inf],[4,0,6,14,inf],[2,6,0,8,inf],[10,14,8,0,inf],[inf,inf,inf,inf,0]]))

if __name__ == '__main__':
    TestBrain.main()