import numpy as np
from typing import List
from utils import validate_binary, validate_loopless, validate_square

class PreferredPath():

    # Static variables

    _DEF_METHOD = 'fwd'
    _DEF_OUT_PATH = False


    # Constructor

    def __init__(self, adj: np.ndarray, fn_vector: List['function'], fn_weights: List[float], validate: bool = True):
        """
        Computes paths between node pairs, allowing for multiple criteria to select the next location at any step, as well as weighting of these criteria

        Parameters
        ----------
        adj : np.ndarray
            Binary adjacency matrix that defines where edges are present
        fn_vector : List[function]
            Sequence of functions used to give each node a score as the next node in the path, with parameters:
            - loc : int
                Current node
            - nxt : int
                Next node
            - prev_nodes : list
                Path sequence so far (excluding 'loc')
            - target : int
                Target node
        fn_weights : List[float]
            Sequence of weights used to weight the importance of each function
        validate : bool, optional
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
    def fn_length(self) -> int:
        """
        Get the number of functions used as criteria in the model
        """
        return self._fn_len

    @property
    def fn_vector(self) -> List['function']:
        """
        Get the list of functions used as criteria in the model
        """
        return self._fn_vector

    @property
    def fn_weights(self) -> List[int]:
        """
        Get the list of weights applied to each function in the model
        """
        return self._fn_weights

    @fn_weights.setter
    def fn_weights(self, value: List[float]) -> None:
        """
        Sets the list of weights applied to each function in the model

        Parameters
        ----------
        value : List[float]
            List of weights
        """

        if self._validate:
            new_len = len(value)
            if new_len != self.fn_length:
                raise ValueError(f"The length of the list of weights ({new_len}) differs to the length of the list of functions ({self.fn_length})")
        self._fn_weights = value

    @property
    def adj(self) -> np.ndarray:
        """
        Get the adjacency matrix that defines where edges are present
        """
        return self._adj

    @adj.setter
    def adj(self, value: np.ndarray) -> None:
        """
        Sets the adjacency matrix that defines where edges are present

        Parameters
        ----------
        value : np.ndarray
            Binary adjacency matrix that defines where edges are present
        """

        if self._validate:
            validate_square(value)
            validate_binary(value)
            validate_loopless(value)
        self._adj = value


    # Methods

    def retrieve_all_paths(self, method: str = _DEF_METHOD, out_path: bool =_DEF_OUT_PATH) -> 'dict or np.ndarray':
        """
        Returns the path sequences for all source and target nodes

        Parameters
        ----------
        method : str, optional
            Path navigation type, by default 'fwd'
            'rev'  : Revisits allowed. If a revisit occurs, that the path sequence equals 'None' due to entering an infinite loop
            'fwd'  : Forward only, nodes cannot be revisited and backtracking isn't allowed
            'back' : Backtracking allowed, nodes cannot be revisited and backtracking to previous nodes occur at dead ends to find alternate routes
        out_path : bool, optional
            Whether to output the full path sequence (True) or the number of hops in each path sequence (False), by default False

        Returns
        -------
        dict or numpy.ndarray
            Path sequences as a dict (out_path=True) or path lengths as a matrix (out_path=False) for all node pairs.
            E.g. out[1][4] = path sequence or path length for source node 1 and target node 4)
        """

        fn = self._convert_method_to_fn(method) # e.g. forward only
        M = self._path_dict(self._res) if out_path else np.zeros((self._res, self._res))
        for source in range(self._res):
            for target in range(self._res):
                if source != target:
                    M[source][target] = PreferredPath._single_path_formatted(fn, source, target, out_path)
        return M

    def retrieve_single_path(self, source: int, target: int, method: str = _DEF_METHOD, out_path: bool = _DEF_OUT_PATH) -> 'List[int] or int':
        """
        Returns the preferred path sequence for a single source and target node

        Parameters
        ----------
        source : int
            Source node
        target : int
            Target node
        method : str, optional
            Path navigation type, by default 'fwd'
            'rev'  : Revisits allowed. If a revisit occurs, that the path sequence equals 'None' due to entering an infinite loop
            'fwd'  : Forward only, nodes cannot be revisited and backtracking isn't allowed
            'back' : Backtracking allowed, nodes cannot be revisited and backtracking to previous nodes occur at dead ends to find alternate routes
        out_path : bool, optional
            Whether to output the full path sequence (True) or the number of hops in each the sequence (False), by default False

        Returns
        -------
        List[int] or int
            Path sequence as a list if successful (out_path=True) or path length as an int (out_path=False) for the given node pair
        """

        fn = self._convert_method_to_fn(method)
        return PreferredPath._single_path_formatted(fn, source, target, out_path)


    # Internal

    @staticmethod
    def _single_path_formatted(fn, source: int, target: int, out_path: bool) -> 'List[int] or int':
        """
        Returns the preferred path formatted as the path length or path sequence, depending on the out_path boolean.
        Used by retrieve_all_paths and retrieve_single_path

        Parameters
        ----------
        fn : function
            Path algorithm method to use (_fwd, _rev or _back)
        source : int
            Source node
        target : int
            Target node
        out_path : bool
            Whether to output the full path sequence (True) or the number of hops in the path sequence (False)

        Returns
        -------
        List[int] or int
            Path sequence as a list if successful (out_path=True) or path length as an int (out_path=False) for the given node pair
        """

        path = fn(source, target)
        if out_path: return path
        elif path is not None: return len(path) - 1
        else: return -1

    def _fwd(self, source: int, target: int) -> List[int]:
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
        List[int]
            Path sequence for the given node pair if successful
        """

        loc = source
        prev = []
        remaining = np.full(self._res, True)
        while loc != target:
            remaining[loc] = False
            nxt = self._next_loc_fn(source, target, loc, prev, remaining)
            if nxt is None:
                return None
            prev.append(loc)
            loc = nxt
        prev.append(target)
        return prev

    def _rev(self, source: int, target: int) -> List[int]:
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
        List[int]
            Path sequence for the given node pair if successful
        """

        loc = source
        prev = []
        remaining = np.full(self._res, True)
        while loc != target:
            nxt = self._next_loc_fn(source, target, loc, prev, remaining)
            if nxt is None or nxt in prev:
                return None
            prev.append(loc)
            loc = nxt
        prev.append(target)
        return prev

    def _back(self, source: int, target: int) -> List[int]:
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
        List[int]
            Path sequence for the given node pair if successful
        """

        loc = source
        prev = []
        remaining = np.full(self._res, True)
        while loc != target:
            remaining[loc] = False
            nxt = self._next_loc_fn(source, target, loc, prev, remaining)
            if nxt is not None: prev.append(loc)
            elif prev: nxt = prev.pop() # Backtrack here
            else: return None # Nowhere to backtrack (graph is disconnected)
            loc = nxt
        prev.append(loc)
        return prev

    def _next_loc_fn(self, source: int, target: int, loc: int, prev: List[int], remaining: np.ndarray) -> int:
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
        prev : List[int]
            Path sequence so far (excluding 'loc')
        remaining : np.ndarray
            Boolean vector containing which nodes can be chosen from

        Returns
        -------
        int
            Next location
        """

        candidates = np.where((remaining == True) & (self._adj[loc] == 1))[0]
        if candidates.size == 0:
            return None
        total_scores = self._get_total_scores(loc, candidates, prev, target)
        total_max = total_scores.max()
        mask = np.where(total_scores == total_max)
        best_cand = candidates[mask]
        if best_cand.size == 1:
            return best_cand[0]
        return np.random.choice(best_cand)

    def _get_total_scores(self, loc: int, candidates: np.ndarray, prev: List[int], target: int) -> np.ndarray:
        """
        Returns the overall scores for nodes as the next location in a preferred path

        Parameters
        ----------
        loc : int
            Current location
        candidates : np.ndarray
            Nodes that can be selected as the next location
        prev : List[int]
            Path sequence so far (excluding 'loc')
        target : int
            Target node (last node in the path, not necessarily in 'candidates')

        Returns
        -------
        np.ndarray
            Overall scores for each potential next node
        """

        num_cand = candidates.size
        total_scores = np.zeros(num_cand)
        for i in range(self._fn_len):
            temp_score = PreferredPath._get_temp_scores(self._fn_vector[i], num_cand, loc, candidates, prev, target)
            total_scores += self._fn_weights[i] * temp_score
        return total_scores

    @staticmethod
    def _get_temp_scores(fn: 'function', num_cand: int, loc: int, candidates: np.ndarray, prev: List[int], target: int) -> np.ndarray:
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
        candidates : np.ndarray
            Nodes that can be selected as the next location
        prev : List[int]
            Path sequence so far (excluding 'loc')
        target : int
            Target node (last node in the path, not necessarily in 'candidates')

        Returns
        -------
        np.ndarray
            Vector of node scores for a single function
        """

        scores = np.zeros(num_cand)
        for i in range(num_cand):
            scores[i] = fn(loc, candidates[i], prev, target)
        score_max = abs(scores).max()
        if score_max != 0:
            return scores / score_max
        return scores

    def _convert_method_to_fn(self, method: str) -> 'function':
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
        function
            Relevant function for the given method

        Raises
        ------
        ValueError
            If method name is invalid
        """

        if method == 'fwd': return self._fwd
        elif method == 'rev': return self._rev
        elif method == 'back': return self._back
        else: raise ValueError("Invalid method")

    @staticmethod
    def _path_dict(n: int) -> dict:
        """
        Returns an empty two-layered dictionary with keys for each source node and empty dictionaries as values

        Parameters
        ----------
        n : int
            Number of nodes

        Returns
        -------
        dict
            Two-layered empty path dictionary
        """

        M = {}
        for i in range(n):
            M[i] = {}
        return M