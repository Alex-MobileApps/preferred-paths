# cd to package top-level directory
# run: python3 -m unittest test.test_preferred_path.py

from test.test import Test
from preferred_path import PreferredPath
import numpy as np

class TestPaths(Test):

    @classmethod
    def setUp(cls):
        cls.sc = np.array(
            [[0,5,0,0,0,0,0,0],
             [5,0,4,0,0,9,0,0],
             [0,4,0,0,0,1,7,0],
             [0,0,0,0,2,6,7,0],
             [0,0,0,2,0,3,0,0],
             [0,9,1,6,3,0,0,0],
             [0,0,7,7,0,0,0,0],
             [0,0,0,0,0,0,0,0]])
        cls.adj = (cls.sc > 0).astype(np.int)
        cls.deg = cls.adj.sum(axis=0)
        cls.fn_vector = [lambda s, t, prev: cls.sc[s,t], lambda s, t, prev: cls.deg[t]]
        cls.fn_weights = [0.4, 0.7]
        cls.pp1 = PreferredPath(cls.adj, cls.fn_vector, cls.fn_weights)

    def test_init(self):
        np.testing.assert_equal(self.pp1._adj, self.adj)
        self.assertEqual(self.pp1._res, len(self.sc))
        self.assertEqual(self.pp1._fn_vector, self.fn_vector)
        self.assertEqual(self.pp1._fn_weights, self.fn_weights)
        self.assertEqual(self.pp1._num_fns, len(self.fn_weights))
        test = lambda arr, r=True: self.assert_raise(r, lambda: PreferredPath(arr, [], [], validate=r))
        test(np.array([[0,0]]), False)   # No validation
        test(np.array([[0,0]]))          # Non-square
        test(np.array([[0,0]]))          # Non-square
        test(np.array([[0,0],[1,0]]))    # Directed
        test(np.array([[0,2],[2,0]]))    # Non-binary
        test(np.array([[1,1],[1,0]]))    # Non-loopless
        test = lambda fn_vector, fn_weights, r=True: self.assert_raise(r, lambda: PreferredPath(np.array([[0]]), fn_vector, fn_weights, validate=r))
        test([], [1], False)             # No validation
        test([], [1])                    # Unequal shapes
        test([lambda s, t, prev: s], []) # Unequal shapes

    def test_retrieve_all_paths(self):
        fn = self.pp1.retrieve_all_paths
        self.assert_raise(True, lambda: fn('a')) # Invalid method
        test = lambda method, res: self.assertDictEqual(fn(method), res)

        # Forward
        res_fwd = {
            0: {1: [0,1], 2: [0,1,5,3,6,2], 3: [0,1,5,3], 4: None, 5: [0,1,5], 6: [0,1,5,3,6], 7: None},
            1: {2: [1,5,3,6,2], 3: [1,5,3], 4: None, 5: [1,5], 6: [1,5,3,6], 7: None},
            2: {3: None, 4: None, 5: [2,5], 6: None, 7: None},
            3: {4: None, 5: [3,5], 6: [3,5,1,2,6], 7: None},
            4: {5: [4,5], 6: [4,5,1,2,6], 7: None},
            5: {6: [5,1,2,6], 7: None},
            6: {7: None}}
        test('fwd', res_fwd)

        # Revisits
        res_rev = {
            0: {1: [0,1], 2: None, 3: None, 4: None, 5: [0,1,5], 6: None, 7: None},
            1: {2: None, 3: None, 4: None, 5: [1,5], 6: None, 7: None},
            2: {3: None, 4: None, 5: [2,5], 6: None, 7: None},
            3: {4: None, 5: [3,5], 6: None, 7: None},
            4: {5: [4,5], 6: None, 7: None},
            5: {6: None, 7: None},
            6: {7: None}}
        test('rev', res_rev)

        # Backtrack
        res_back = {
            0: {1: [0,1], 2: [0,1,5,3,6,2], 3: [0,1,5,3], 4: [0,1,5,3,4], 5: [0,1,5], 6: [0,1,5,3,6], 7: None},
            1: {2: [1,5,3,6,2], 3: [1,5,3], 4: [1,5,3,4], 5: [1,5], 6: [1,5,3,6], 7: None},
            2: {3: [2,5,3], 4: [2,5,3,4], 5: [2,5], 6: [2,5,3,6], 7: None},
            3: {4: [3,5,4], 5: [3,5], 6: [3,5,1,2,6], 7: None},
            4: {5: [4,5], 6: [4,5,1,2,6], 7: None},
            5: {6: [5,1,2,6], 7: None},
            6: {7: None}}
        test('back', res_back)

    def test_retrieve_single_path(self):
        fn = self.pp1.retrieve_single_path
        self.assert_raise(True, lambda: fn(0, 1, 'a')) # Invalid method
        test = lambda method, source, target, res: self.assertEqual(fn(source, target, method), res)
        test('fwd', 0, 5, [0,1,5])    # Success
        test('fwd', 0, 3, [0,1,5,3])  # Success when 'rev' fails
        test('fwd', 3, 0, None)       # Failed: Needs backtracking
        test('fwd', 0, 7, None)       # Failed: Isolated node
        test('rev', 0, 5, [0,1,5])    # Success
        test('rev', 0, 3, None)       # Failed: Revisited
        test('rev', 3, 0, None)       # Failed: Revisited
        test('rev', 0, 7, None)       # Failed: Isolated node
        test('back', 0, 5, [0,1,5])   # Success
        test('back', 0, 3, [0,1,5,3]) # Success when 'rev' fails
        test('back', 3, 0, [3,5,1,0]) # Success when 'fwd' fails
        test('back', 0, 7, None)      # Failed: Isolated node

    def test_fwd(self):
        test = lambda source, target, res: self.assertEqual(self.pp1._fwd(source, target), res)
        test(0, 5, [0,1,5])   # Success
        test(0, 3, [0,1,5,3]) # Success when 'rev' fails
        test(3, 0, None)      # Failed: Needs backtracking
        test(0, 7, None)      # Failed: Isolated node

    def test_rev(self):
        test = lambda source, target, res: self.assertEqual(self.pp1._rev(source, target), res)
        test(0, 5, [0,1,5]) # Success
        test(0, 3, None)    # Failed: Revisited
        test(3, 0, None)    # Failed: Revisited
        test(0, 7, None)    # Failed: Isolated node

    def test_back(self):
        test = lambda source, target, res: self.assertEqual(self.pp1._back(source, target), res)
        test(0, 5, [0,1,5])   # Success
        test(0, 3, [0,1,5,3]) # Success when 'rev' fails
        test(3, 0, [3,5,1,0]) # Success when 'fwd' fails
        test(0, 7, None)      # Failed: Isolated node

    def test_next_loc_fn(self):
        # No remaining candidates
        remaining = np.full(self.pp1._res, False)
        test = lambda loc, res, prev: np.testing.assert_equal(self.pp1._next_loc_fn(loc, prev, remaining), res)
        test(0, None, [])

        # Revists allowed
        remaining = np.full(self.pp1._res, True)
        test(3, 5, [5])

        # test random choices
        res = [False, False]
        for _ in range(20):
            temp = self.pp1._next_loc_fn(6, [], remaining)
            if temp == 2: res[0] = True
            elif temp == 3: res[1] = True
            else: self.fail("Random choice needs to be 2 or 3")
        self.assertEqual(res, [True, True])

        # No revisit path
        remaining[0] = False
        test(0, 1, [])
        remaining[1] = False
        test(1, 5, [0])
        remaining[5] = False
        test(5, 3, [0,1])

    def test_get_total_scores(self):
        fn = self.pp1._get_total_scores
        exp = fn(0, np.array([1]), [])
        self.assert_float(exp, np.array([1.1]))

        exp = fn(1, np.array([0,2,5]), [0]) # if revisits
        self.assert_float(exp, np.array([.4, .7, 1.1]))

        exp = fn(5, np.array([2,3,4]), [0,1])
        self.assert_float(exp, np.array([.77, 1.1, .67]))

    def test_get_temp_scores(self):
        fn1 = self.pp1._fn_vector[0]
        fn2 = self.pp1._fn_vector[1]

        exp = lambda fn: PreferredPath._get_temp_scores(fn, 1, 0, np.array([1]), [])
        test = lambda fn, res: self.assert_float(exp(fn), res)
        test(fn1, np.array([1]))
        test(fn2, np.array([1]))

        exp = lambda fn: PreferredPath._get_temp_scores(fn, 2, 1, np.array([2,5]), [0])
        test(fn1, np.array([.44,1]))
        test(fn2, np.array([ .75,1]))

        exp = lambda fn: PreferredPath._get_temp_scores(fn, 3, 5, np.array([2,3,4]), [0,1])
        test(fn1, np.array([.17,   1,   .5]))
        test(fn2, np.array([  1,   1,  .67]))

        exp = lambda fn: PreferredPath._get_temp_scores(fn, 1, 0, np.array([7]), [])
        test(fn2, np.array([0])) # Divide by zero handled

    def test_convert_method_to_fn(self):
        self.assertEqual(self.pp1._convert_method_to_fn('fwd'), self.pp1._fwd)
        self.assertEqual(self.pp1._convert_method_to_fn('rev'), self.pp1._rev)
        self.assertEqual(self.pp1._convert_method_to_fn('back'), self.pp1._back)
        self.assert_raise(True, lambda: self.pp1._convert_method_to_fn('a'))

    def test_path_list(self):
        expected3 = {
            0:{1:None,2:None},
            1:{2:None}}
        expected5 = {
            0:{1:None,2:None,3:None,4:None},
            1:{2:None,3:None,4:None},
            2:{3:None,4:None},
            3:{4:None}}
        self.assertDictEqual(PreferredPath._path_list(3), expected3)
        self.assertDictEqual(PreferredPath._path_list(5), expected5)

if __name__ == '__main__':
    TestPaths.main()