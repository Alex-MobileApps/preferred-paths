import numpy as np

from utils import validate_binary, validate_loopless, validate_square

class PreferredPath():

    # Static variables

    _DEF_METHOD = 'fwd'
    _DEF_OUT_PATH = False


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
            Whether or not to validate parameter changes when optimising
        """

        self._validate = validate
        self.adj = adj
        self._res = len(self.adj)
        self._fn_len = len(fn_vector)
        self._fn_vector = fn_vector
        self.fn_weights = fn_weights


    # Properties

    @property
    def fn_length(self):
        """
        Get the number of functions used as criteria in the model
        """
        return self._fn_len

    @property
    def fn_vector(self):
        """
        Get the list of functions used as criteria in the model
        """
        return self._fn_vector

    @property
    def fn_weights(self):
        """
        Get the list of weights applied to each function in the model
        """
        return self._fn_weights

    @fn_weights.setter
    def fn_weights(self, value):
        """
        Sets the list of weights applied to each function in the model

        Parameters
        ----------
        value : list
            List of weights
        """

        if self._validate:
            new_len = len(value)
            if new_len != self.fn_length:
                raise ValueError(f"The length of the list of weights ({new_len}) differs to the length of the list of functions ({self.fn_length})")
        self._fn_weights = value

    @property
    def adj(self):
        """
        Get the adjacency matrix that defines where edges are present
        """
        return self._adj

    @adj.setter
    def adj(self, value):
        """
        Sets the adjacency matrix that defines where edges are present

        Parameters
        ----------
        value : numpy.ndarray
            Binary adjacency matrix that defines where edges are present
        """

        if self._validate:
            validate_square(value)
            validate_binary(value)
            validate_loopless(value)
        self._adj = value


    # Methods

    def retrieve_all_paths(self, method=_DEF_METHOD, out_path=_DEF_OUT_PATH):
        """
        Returns the path sequences for all source and target nodes

        Parameters
        ----------
        method : str
            'rev'  : Revisits allowed. If a revisit occurs, that the path sequence equals 'None' due to entering an infinite loop
            'fwd'  : Forward only, nodes cannot be revisited and backtracking isn't allowed
            'back' : Backtracking allowed, nodes cannot be revisited and backtracking to previous nodes occur at dead ends to find alternate routes
        out_path : bool
            Whether to output the full path sequence (True) or the number of hops in each path sequence (False)

        Returns
        -------
        out : dict or numpy.ndarray
            Path sequences as a dict (out_path=True) or path lengths as a matrix (out_path=False) for all node pairs.
            E.g. out[1][4] = path sequence or path length for source node 1 and target node 4)
        """

        fn = self._convert_method_to_fn(method)
        M = self._path_dict(self._res) if out_path else np.zeros((self._res, self._res))
        for source in range(self._res):
            for target in range(self._res):
                if source != target:
                    M[source][target] = PreferredPath._single_path_formatted(fn, source, target, out_path)
        return M

    def retrieve_single_path(self, source, target, method=_DEF_METHOD, out_path=_DEF_OUT_PATH):
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
        out_path : bool
            Whether to output the full path sequence (True) or the number of hops in each the sequence (False)

        Returns
        -------
        out : list or int
            Path sequence as a list if successful (out_path=True) or path length as an int (out_path=False) for the given node pair
        """

        fn = self._convert_method_to_fn(method)
        return PreferredPath._single_path_formatted(fn, source, target, out_path)


    # Internal

    @staticmethod
    def _single_path_formatted(fn, source, target, out_path):
        """
        Returns the preferred path formatted as the path length or path sequence, depending on the out_path boolean.
        Used by retrieve_all_paths and retrieve_single_path

        Parameters
        ----------
        source : int
            Source node
        target : int
            Target node
        out_path : bool
            Whether to output the full path sequence (True) or the number of hops in the path sequence (False)

        Returns
        -------
        out : list or int
            Path sequence as a list if successful (out_path=True) or path length as an int (out_path=False) for the given node pair
        """

        path = fn(source, target)
        if out_path: return path
        elif path is not None: return len(path) - 1
        else: return np.inf

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
            next_loc = self._next_loc_fn(source, target, loc, prev, remaining)
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
            next_loc = self._next_loc_fn(source, target, loc, prev, remaining)
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
            next_loc = self._next_loc_fn(source, target, loc, prev, remaining)
            if next_loc is not None: prev.append(loc)
            elif prev: next_loc = prev.pop() # Backtrack here
            else: return None # Nowhere to backtrack (graph is disconnected)
            loc = next_loc
        prev.append(loc)
        return prev

    def _next_loc_fn(self, source, target, loc, prev, remaining):
        """
        Returns the next location (node) in a preferred path

        Parameters
        ----------
        source : int
            Source node
        target : int
            Target node
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

        candidates = np.argwhere((remaining == True) & (self._adj[loc] == 1)).ravel()
        if candidates.size == 0:
            return None
        total_scores = self._get_total_scores(loc, candidates, prev)
        best_cand = candidates[np.argwhere(total_scores == total_scores.max()).ravel()]
        choice = np.random.choice(best_cand)
        if len(best_cand) > 1:
            print(f"Warning - Path {source}-{target} at node {loc}: Multiple candidate nodes found. Randomly selecting node {choice} from {best_cand} (previous: {prev})")
        return choice

    def _get_total_scores(self, loc, candidates, prev):
        """
        Returns the overall scores for nodes as the next location in a preferred path

        Parameters
        ----------
        loc : int
            Current location
        candidates : numpy.ndarray
            Nodes that can be selected as the next location
        prev : list
            Path sequence so far (excluding 'loc')
        """

        num_cand = len(candidates)
        total_scores = np.zeros(num_cand)
        for i in range(self._fn_len):
            temp_score = PreferredPath._get_temp_scores(self._fn_vector[i], num_cand, loc, candidates, prev)
            total_scores += self._fn_weights[i] * temp_score
        return total_scores

    @staticmethod
    def _get_temp_scores(fn, num_cand, loc, candidates, prev):
        """
        Returns the single function score for nodes as the next location in a preferred path

        Parameters
        ----------
        fn : function
            Function used to compute a node score, with parameters - source: int, target: int, prev: list
        num_cand : int
            Number of nodes to choose from for the next location
        loc : int
            Current location
        candidates : numpy.ndarray
            Nodes that can be selected as the next location
        prev : list
            Path sequence so far (excluding 'loc')

        Returns
        -------
        out : numpy.ndarray
            Vector of node scores for a single function
        """

        scores = np.zeros(num_cand)
        for i in range(num_cand):
            scores[i] = fn(loc, candidates[i], prev)
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
    def _path_dict(n):
        """
        Returns a two-layered dictionary with keys for each source node and empty dictionaries as values

        Parameters
        ----------
        n : int
            Number of nodes

        Returns
        -------
        out : dict
            Two-layered dictionary
        """

        M = {}
        for i in range(n):
            M[i] = {}
        return M