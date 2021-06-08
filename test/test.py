from unittest import TestCase
import numpy as np

class Test(TestCase):

    def assert_float(self, result, expected):
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-2, verbose=True)

    def assert_raise(self, expected, fn, *args, **kwargs):
        result = False
        try: fn(*args, **kwargs)
        except: result = True
        self.assertEqual(result, expected)

    def setup_unw(cls):
        cls.fc_bin1 = np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
        cls.sc_bin1 = np.array([[0,0,0,0,0],[0,0,1,1,1],[0,1,0,1,0],[0,1,1,0,1],[0,1,0,1,0]])
        cls.sc_bin2 = np.array([[0,0,1,1,1],[0,0,1,1,1],[1,1,0,1,0],[1,1,1,0,1],[1,1,0,1,0]])
        cls.sc_bin3 = np.array([[0,1,1,1,1],[1,0,1,1,1],[1,1,0,1,0],[1,1,1,0,1],[1,1,0,1,0]])
        cls.sc_bin4 = np.array([[0,0,1,1,1],[0,0,0,1,1],[1,0,0,1,0],[1,1,1,0,1],[1,1,0,1,0]])
        cls.sc_bin5 = np.array([[0,0,1,1,1],[0,0,0,0,1],[1,0,0,1,0],[1,0,1,0,1],[1,1,0,1,0]])
        cls.sc_bin6 = np.array([[0,0,1,1,0],[0,0,0,0,1],[1,0,0,1,0],[1,0,1,0,1],[0,1,0,1,0]])

    def setup_wei(cls):
        cls.sc1 = np.array([[0,1,2,3,4,5,16],[1,0,6,7,8,17,18],[2,6,0,9,10,19,20],[3,7,9,0,11,21,12],[4,8,10,11,0,22,13],[5,17,19,21,22,0,14],[16,18,20,12,13,14,0]])
        cls.sc2 = np.array([[0,1,2,14,3,4],[1,0,15,5,11,12],[2,15,0,6,7,8],[14,5,6,0,10,13],[3,11,7,10,0,9],[4,12,8,13,9,0]])
        cls.sc_bin1 = np.array([[0,0,0,0,0,0,1],[0,0,0,0,0,1,1],[0,0,0,0,0,1,1],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0],[0,1,1,1,1,0,0],[1,1,1,0,0,0,0]])
        cls.sc_bin2 = np.array([[0,0,0,1,0,0],[0,0,1,0,1,1],[0,1,0,0,0,0],[1,0,0,0,1,1],[0,1,0,1,0,0],[0,1,0,1,0,0]])
        cls.fc_bin1 = np.array([[0,0,1,0,0,0,0],[0,0,0,1,0,0,1],[1,0,0,0,0,1,0],[0,1,0,0,0,0,0],[0,0,0,0,0,1,0],[0,0,1,0,1,0,1],[0,1,0,0,0,1,0]])
        cls.fc_bin2 = np.array([[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]])
        cls.fill1 = lambda cls, method: (cls.sc1, cls.sc_bin1, cls.fc_bin1, method)

    def setup_brain(cls):
        load = lambda txt: np.loadtxt(f'test/test_{txt}.txt', delimiter=',')
        cls.sc1 = load('sc1')
        cls.sc2 = load('sc2')
        cls.fc1 = load('fc1')
        cls.fc2 = load('fc2')
        cls.sc_thresh1 = 1
        cls.sc_thresh2 = 5
        cls.fc_thresh1 = 0.01
        cls.fc_thresh2 = 0.1
        cls.sc_thresh_type1 = 'pos'
        cls.sc_thresh_type2 = 'neg'
        cls.fc_thresh_type1 = 'pos'
        cls.fc_thresh_type2 = 'abs'
        cls.res = 20