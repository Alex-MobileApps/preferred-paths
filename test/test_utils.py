# cd to package top-level directory
# run: python3 -m unittest test.test_utils.py

import numpy as np
import utils
from test.test import Test

class TestUtils(Test):

    @classmethod
    def setUp(cls):
        cls.setup_unw(cls)

    def test_binarise_matrix(self):
        M = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
        test = lambda thresh_type, res: self.assert_float(utils.binarise_matrix(M, -2, thresh_type), res)
        test('pos', np.array([0,0,0,1,1,1,1,1,1,1,1]))
        test('neg', np.array([1,1,1,1,0,0,0,0,0,0,0]))
        test('abs', np.array([1,1,1,1,0,0,0,1,1,1,1]))

    def test_validate_binary(self):
        fn = utils.validate_binary
        bin = np.array([[0,1],[1,0]])
        wei = np.array([[0,1],[2,3]])
        self.assert_raise(False, fn, bin)      # x1, binary
        self.assert_raise(True, fn, wei)       # x1, non-binary
        self.assert_raise(False, fn, bin, bin) # x2, binary
        self.assert_raise(True, fn, bin, wei)  # x2, inc. non-binary

    def test_validate_loopless(self):
        fn = utils.validate_loopless
        no_loop = np.array([[0,1],[1,0]])
        loop = np.array([[0,1],[2,3]])
        self.assert_raise(False, fn, no_loop)
        self.assert_raise(True, fn, loop)
        self.assert_raise(False, fn, no_loop, no_loop)
        self.assert_raise(True, fn, no_loop, loop)

    def test_validate_square(self):
        fn = utils.validate_square
        square = np.array([[0,1],[1,0]])
        non_square = np.array([[0,1,2],[1,0,2]])
        self.assert_raise(False, fn, square)            # x1, square
        self.assert_raise(True, fn, non_square)         # x1, non-square
        self.assert_raise(False, fn, square, square)    # x2, square
        self.assert_raise(True, fn, square, non_square) # x2, inc. non-square

    def test_validate_symmetric(self):
        fn = utils.validate_symmetric
        sym = np.array([[0,1],[1,0]])
        non_sym = np.array([[0,1],[2,0]])
        self.assert_raise(False, fn, sym)         # x1, symmetric
        self.assert_raise(True, fn, non_sym)      # x1, non-symmetric
        self.assert_raise(False, fn, sym, sym)    # x2, symmetric
        self.assert_raise(True, fn, sym, non_sym) # x2, inc. non-symmetric

if __name__ == '__main__':
    TestUtils.main()