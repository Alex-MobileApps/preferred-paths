import numpy as np
from itertools import combinations

from utils import validate_binary, validate_loopless, validate_square, validate_symmetric

class PreferredPath():

    # Static variables

    _DEF_METHOD = 'fwd'


    # Constructor

    def __init__(self, adj, fn_vector, fn_weights, validate=True):
        """
        Computes paths between node pairs, allowing for multiple criteria to select the next location at any step, as well as weighting of these criteria

        Parameters
        ----------
        adj : numpy.ndarray
            Binary adjacency matrix that defines where edges are present
        fn_vector : list
            Sequence of functions used to give each node a score as the next node in the path, with parameters:
            - source : int
                current node
            - target : int
                Next node
            - prev : list
                Path sequence so far (excluding 'source')
        fn_weights : list
            Sequence of weights used to weight the importance of each function
        validate : bool
            Whether or not to validate arguments on initialisation
        """

        self._num_fns = len(fn_vector)

        if validate:
            validate_square(adj)
            validate_symmetric(adj)
            validate_binary(adj)
            validate_loopless(adj)
            if self._num_fns != len(fn_weights):
                raise ValueError("Feature vector and weights must be the same length")

        self._adj = adj
        self._res = len(adj)
        self._fn_vector = fn_vector
        self._fn_weights = fn_weights


    # Methods

    def retrieve_all_paths(self, method=_DEF_METHOD):
        """
        Returns the path sequences for all source and target nodes

        Parameters
        ----------
        method : str
            'rev'  : Revisits allowed. If a revisit occurs, that the path sequence equals 'None' due to entering an infinite loop
            'fwd'  : Forward only, nodes cannot be revisited and backtracking isn't allowed
            'back' : Backtracking allowed, nodes cannot be revisited and backtracking to previous nodes occur at dead ends to find alternate routes

        Returns
        -------
        out : dict
            Path sequences for all node pairs (e.g. out[1][4] = path sequence for source node 1 and target node 4)
        """

        fn = self._convert_method_to_fn(method)
        M = PreferredPath._path_list(self._res)
        for source in range(self._res - 1):
            for target in range(source + 1, self._res):
                M[source][target] = fn(source, target)
        return M

    def retrieve_single_path(self, source, target, method=_DEF_METHOD):
        """
        Returns the preferred path sequence for a single source and target node

        Parameters
        ----------
        source : int
            Source node
        target : int
            Target node
        method : str
            'rev'  : Revisits allowed. If a revisit occurs, that the path sequence equals 'None' due to entering an infinite loop
            'fwd'  : Forward only, nodes cannot be revisited and backtracking isn't allowed
            'back' : Backtracking allowed, nodes cannot be revisited and backtracking to previous nodes occur at dead ends to find alternate routes

        Returns
        -------
        out : list
            Path sequence for the given node pair if successful
        """

        fn = self._convert_method_to_fn(method)
        return fn(source, target)


    # Internal

    def _fwd(self, source, target):
        """
        Returns the preferred path sequence for a single source and target node using the 'forward only' method

        Parameters
        ----------
        source : int
            Source node
        target : int
            Target node

        Returns
        -------
        out : list
            Path sequence for the given node pair if successful
        """

        loc = source
        prev = []
        remaining = np.full(self._res, True)
        while loc != target:
            remaining[loc] = False
            next_loc = self._next_loc_fn(loc, prev, remaining)
            if next_loc is None:
                return None
            prev.append(loc)
            loc = next_loc
        prev.append(target)
        return prev

    def _rev(self, source, target):
        """
        Returns the preferred path sequence for a single source and target node using the 'revisits' method

        Parameters
        ----------
        source : int
            Source node
        target : int
            Target node

        Returns
        -------
        out : list
            Path sequence for the given node pair if successful
        """

        loc = source
        prev = []
        remaining = np.full(self._res, True)
        while loc != target:
            next_loc = self._next_loc_fn(loc, prev, remaining)
            if next_loc is None or next_loc in prev:
                return None
            prev.append(loc)
            loc = next_loc
        prev.append(target)
        return prev

    def _back(self, source, target):
        """
        Returns the preferred path sequence for a single source and target node using the 'backtracking allowed' method

        Parameters
        ----------
        source : int
            Source node
        target : int
            Target node

        Returns
        -------
        out : list
            Path sequence for the given node pair if successful
        """

        loc = source
        prev = []
        remaining = np.full(self._res, True)
        while loc != target:
            remaining[loc] = False
            next_loc = self._next_loc_fn(loc, prev, remaining)
            if next_loc is not None: prev.append(loc)
            elif prev: next_loc = prev.pop() # Backtrack here
            else: return None # Nowhere to backtrack (graph is disconnected)
            loc = next_loc
        prev.append(loc)
        return prev

    def _next_loc_fn(self, loc, prev, remaining):
        """
        Returns the next location (node) in a preferred path

        Parameters
        ----------
        loc : int
            Current location
        prev : list
            Path sequence so far (excluding 'loc')
        remaining : numpy.ndarray
            Boolean vector containing which nodes can be chosen from

        Returns
        -------
        out : int
            Next location
        """

        targets = np.argwhere((remaining == True) & (self._adj[loc] == 1)).ravel()
        if targets.size == 0:
            return None
        total_scores = self._get_total_scores(loc, targets, prev)
        candidates = targets[np.argwhere(total_scores == total_scores.max()).ravel()]
        return np.random.choice(candidates)

    def _get_total_scores(self, loc, targets, prev):
        """
        Returns the overall scores for nodes as the next location in a preferred path

        Parameters
        ----------
        loc : int
            Current location
        targets : numpy.ndarray
            Nodes that can be selected as the next location
        prev : list
            Path sequence so far (excluding 'loc')
        """

        num_targets = len(targets)
        total_scores = np.zeros(num_targets)
        for i in range(self._num_fns):
            temp_score = PreferredPath._get_temp_scores(self._fn_vector[i], num_targets, loc, targets, prev)
            total_scores += self._fn_weights[i] * temp_score
        return total_scores

    @staticmethod
    def _get_temp_scores(fn, num_targets, loc, targets, prev):
        """
        Returns the single function score for nodes as the next location in a preferred path

        Parameters
        ----------
        fn : function
            Function used to compute a node score, with parameters - source: int, target: int, prev: list
        num_targets : int
            Number of nodes to choose from for the next location
        loc : int
            Current location
        targets : numpy.ndarray
            Nodes that can be selected as the next location
        prev : list
            Path sequence so far (excluding 'loc')

        Returns
        -------
        out : numpy.ndarray
            Vector of node scores for a single function
        """

        scores = np.zeros(num_targets)
        for i in range(num_targets):
            scores[i] = fn(loc, targets[i], prev)
        score_max = scores.max()
        return scores / score_max if score_max != 0 else scores

    def _convert_method_to_fn(self, method):
        """
        Returns the relevant function for the requested path navigation method

        Parameters
        ----------
        method : str
            'rev'  : Revisits allowed. If a revisit occurs, that the path sequence equals 'None' due to entering an infinite loop
            'fwd'  : Forward only, nodes cannot be revisited and backtracking isn't allowed
            'back' : Backtracking allowed, nodes cannot be revisited and backtracking to previous nodes occur at dead ends to find alternate routes

        Returns
        -------
        out : function
            Relevant function for the given method
        """

        if method == 'fwd': return self._fwd
        elif method == 'rev': return self._rev
        elif method == 'back': return self._back
        else: raise ValueError("Invalid method")

    @staticmethod
    def _path_list(n):
        """
        Returns a dictionary with two layers of keys for each source and target node

        Parameters
        ----------
        n : int
            Number of nodes

        Returns
        -------
        out : dict
            Dictionary with keys for each source and target node
        """

        M = dict()
        for i in range(n - 1):
            M[i] = dict()
        for i, j in combinations(range(n), 2):
            M[i][j] = None
        return M