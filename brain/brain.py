import numpy as np
from utils import binarise_matrix, validate_square, validate_thresh_type
from math import pi, acos
from scipy.sparse.csgraph import dijkstra
from typing import List

class Brain():

   # Static variables

   _DEF_THRESH_TYPE = 'pos'
   _DEF_SC_THRESH = 1
   _DEF_FC_THRESH = 0.01
   _DEF_SC_DIR = False


   # Constructor

   def __init__(self, sc: np.ndarray, fc: np.ndarray, euc_dist: np.ndarray, sc_directed: bool = _DEF_SC_DIR, sc_thresh: float = _DEF_SC_THRESH, fc_thresh: float = _DEF_FC_THRESH, sc_thresh_type: str = _DEF_THRESH_TYPE, fc_thresh_type: str = _DEF_THRESH_TYPE, hubs: np.ndarray = None, regions: np.ndarray = None, func_regions: np.ndarray = None):
      """
      Contains weighted and binarised connectome matrices of a brain, as well as functions that compute features of these connectomes

      Parameters
      ----------
      sc : np.ndarray
         Weighted structural connectivity matrix
      fc : np.ndarray
         Weighted functional connectivity matrix
      euc_dist : np.ndarray
         Euclidean distances between each node pair
      sc_directed : bool, optional
         Whether or not the SC layer is a directed graph or not, by default False
      sc_thresh : float, optional
         Threshold used to binarise SC, by default 1.0
      fc_thresh : float, optional
         Threshold used to binarise FC, by default 0.01
      sc_thresh_type : str, optional
         Method used to binarise the SC matrix, by default 'pos'
         - 'pos' : values less than thresh_val are given 0. Other values are given 1.
         - 'neg' : values greater than thresh_val are given 0. Other values are given 1.
         - 'abs' : absolute values less than the absolute value of thresh_val are given 0. Other values are given 1.
      fc_thresh_type : str, optional
         Method used to binarise the FC matrix, by default 'pos'
         See 'sc_thresh_type' for other values
      hubs : np.ndarray, optional
         Hub node indexes, by default None
      regions : np.ndarray, optional
         Region (as an integer) that each node is assigned to, by default None
      func_regions : np.ndarray, optional
         Functional network (as an integer) that each node is assigned to, by default None
      """

      self._sc_thresh = float(sc_thresh)
      self._fc_thresh = float(fc_thresh)
      validate_thresh_type(sc_thresh_type)
      validate_thresh_type(fc_thresh_type)
      self._sc_thresh_type = sc_thresh_type
      self._fc_thresh_type = fc_thresh_type
      self._sc = Brain._get_wei_matrix(sc)
      self._fc = Brain._get_wei_matrix(fc)
      self._euc_dist = Brain._get_wei_matrix(euc_dist)
      if len(self._sc) != len(self._fc) or len(self._sc) != len(self._euc_dist):
         raise ValueError("Conflicting network resolutions")
      self._sc_bin = Brain._get_bin_matrix(self._sc, self._sc_thresh, self._sc_thresh_type)
      self._fc_bin = Brain._get_bin_matrix(self._fc, self._fc_thresh, self._fc_thresh_type)
      self._sc_directed = sc_directed
      self._sp_hops = None
      self._sp_dist = None

      self._hubs = np.array(hubs, dtype=int) if hubs is not None else np.array([], dtype=int)
      if len(self._hubs) != 0:
         if self._hubs.min() < 0 or self._hubs.max() > len(self._sc) - 1 or len(self._hubs.shape) > 1:
            raise ValueError("Invalid hub node indexes")

      self._regions = Brain._get_regions(regions, self.res)
      self._func_regions = Brain._get_regions(func_regions, self.res)


   # Properties

   @property
   def sc(self) -> np.ndarray:
      """
      Get the weighted SC matrix
      """
      return self._sc

   @property
   def sc_bin(self) -> np.ndarray:
      """
      Get the binarised SC matrix
      """
      return self._sc_bin

   @property
   def sc_thresh(self) -> float:
      """
      Get the threshold used to create the binarised SC matrix
      """
      return self._sc_thresh

   @property
   def sc_thresh_type(self) -> str:
      """
      Get the threshold type used to create the binarised SC matrix
      """
      return self._sc_thresh_type

   @property
   def sc_directed(self) -> bool:
      """
      Get whether or not the SC layer is a directed graph or not
      """
      return self._sc_directed

   @property
   def fc(self) -> np.ndarray:
      """
      Get the weighted FC matrix
      """
      return self._fc

   @property
   def fc_bin(self) -> np.ndarray:
      """
      Get the binarised FC matrix
      """
      return self._fc_bin

   @property
   def fc_thresh(self) -> float:
      """
      Get the threshold used to create the binarised FC matrix
      """
      return self._fc_thresh

   @property
   def fc_thresh_type(self) -> str:
      """
      Get the threshold type used to create the binarised FC matrix
      """
      return self._fc_thresh_type

   @property
   def euc_dist(self) -> np.ndarray:
      """
      Get the Euclidean distances matrix
      """
      return self._euc_dist

   @property
   def res(self) -> int:
      """
      Get the parcellation resolution
      """
      return len(self._sc)


   # Measures in use

   def streamlines(self, weighted: bool = True) -> np.ndarray:
      """
      Returns the number of streamlines between any pair of nodes

      Parameters
      ----------
      weighted : bool, optional
         Whether to use the weighted or unweighted SC matrix, by default True

      Returns
      -------
      np.ndarray
         Matrix of streamlines
      """

      M = self.sc if weighted else self.sc_bin
      return M

   def node_strength(self, weighted: bool = True, method: str = 'tot') -> np.ndarray:
      """
      Returns the sum of edge weights adjacent to a node

      Parameters
      ----------
      weighted : bool, optional
         Whether to use the weighted or unweighted SC matrix, by default True
      method : str, optional
         Used if weighted=True, by default 'tot'
         - 'in'  : in-degree strengths only
         - 'out' : out-degree strengths only
         - 'tot' : combined in and out degree strengths

      Returns
      -------
      np.ndarray
         Vector of strengths for each node
      """

      M = self.sc if weighted else self.sc_bin
      if not self.sc_directed or method == 'out':
         return M.sum(axis=1)
      elif method == 'tot':
         return M.sum(axis=0) + M.sum(axis=1)
      elif method == 'in':
         return M.sum(axis=0)
      else:
         raise ValueError("Invalid method")

   def is_target_node(self, nxt: int, target: int) -> int:
      """
      Returns whether or node the potential next node is the target node

      Parameters
      ----------
      nxt : int
         Next node
      target : int
         Target node

      Returns
      -------
      int
         Whether or not the potential next node is the target node (1 if true, 0 otherwise)
      """

      return int(nxt == target)

   def hubs(self, binary: bool = False) -> np.ndarray:
      """
      Returns the brain's hub nodes

      Parameters
      ----------
      binary : bool, optional
         Whether to return the indexes of the hub nodes (False), or a binary array for all nodes indicating which ones are hubs (True), by default False

      Returns
      -------
      np.ndarray
         Binary vector indicating which nodes are hub nodes
      """

      if not binary:
         return self._hubs
      else:
         M = np.zeros(self.res, dtype=int)
         M[self._hubs] = 1
         return M

   def neighbour_just_visited_node(self, nxt: int, prev_nodes: List[int]) -> int:
      """
      Returns whether or not a potential next node neighbours the most recently visited node

      Parameters
      ----------
      nxt : int
          Next node
      prev_nodes : List[int]
          Path sequence (containing previously visited nodes)

      Returns
      -------
      int
          Whether or not the potential next node neighbours the just visited node (1 if true, 0 otherwise)
      """

      if not prev_nodes:
         return 0
      return self.sc_bin[prev_nodes[-1], nxt]

   # Regions

   def is_target_region(self, nxt: int, target: int) -> int:
      """
      Returns whether or not a potential next node is in the target node's region

      Parameters
      ----------
      nxt : int
          Next node
      target : int
          Target node

      Returns
      -------
      int
          Whether or not the potential next node is the target node's region (1 if true, 0 otherwise)
      """

      return Brain._is_targ_reg(self._regions, nxt, target)

   def edge_con_diff_region(self, loc: int, nxt: int, target: int) -> int:
      """
      Returns whether or not a potential next node leaves the current region, if it is not already in the target region

      Parameters
      ----------
      loc : int
         Current node
      nxt : int
         Next node
      target : int
         Target node

      Returns
      -------
      int
          1 if leaving a non-target region or remaining in the target region, 0 otherwise
      """

      return Brain._edge_con_diff_reg(self._regions, loc, nxt, target)

   def inter_regional_connections(self, weighted: bool = True, distinct: bool = False) -> np.ndarray:
      """
      Returns how many connections each node has to different regions

      Parameters
      ----------
      weighted : bool, optional
          Whether or not to sum the weights of the streamlines of the inter-regional connections, by default True
      distinct : bool, optional
          Whether to count the number of distinct inter-regional connections or the total number of inter-regional connections (only used if weighted=False), by default False

      Returns
      -------
      np.ndarray
          How many inter-regional connections each node has
      """

      return self._int_reg_con(self._regions, weighted, distinct)

   def prev_visited_region(self, loc: int, nxt: int, prev_nodes: List[int]) -> int:
      """
      Returns whether or not the region of a potential next node has already been visited, unless it remains in the same region

      Parameters
      ----------
      loc : int
          Current node
      nxt : int
          Next node
      prev_nodes : List[int]
          Path sequence (containing previously visited nodes)

      Returns
      -------
      int
          Whether or not the region of the potential next has already been visited (1 if already visited and not in the current region, 0 otherwise)
      """

      return Brain._prev_vis_reg(self._regions, loc, nxt, prev_nodes)

   # Functional regions

   def is_target_func_region(self, nxt: int, target: int) -> int:
      """
      Returns whether or not a potential next node is in the target node's functional region

      Parameters
      ----------
      nxt : int
          Next node
      target : int
          Target node

      Returns
      -------
      int
          Whether or not the potential next node is the target node's functional region (1 if true, 0 otherwise)
      """

      return Brain._is_targ_reg(self._func_regions, nxt, target)

   def edge_con_diff_func_region(self, loc: int, nxt: int, target: int) -> int:
      """
      Returns whether or not a potential next node leaves the current functional region, if it is not already in the target functional region

      Parameters
      ----------
      loc : int
         Current node
      nxt : int
         Next node
      target : int
         Target node

      Returns
      -------
      int
          1 if leaving a non-target functional region or remaining in the target functional region, 0 otherwise
      """

      return Brain._edge_con_diff_reg(self._func_regions, loc, nxt, target)

   def prev_visited_func_region(self, loc: int, nxt: int, prev_nodes: List[int]) -> int:
      """
      Returns whether or not the functional region of a potential next node has already been visited, unless it remains in the same functional region

      Parameters
      ----------
      loc : int
          Current node
      nxt : int
          Next node
      prev_nodes : List[int]
          Path sequence (containing previously visited nodes)

      Returns
      -------
      int
          Whether or not the functional region of the potential next has already been visited (1 if already visited and not in the current functional region, 0 otherwise)
      """

      return Brain._prev_vis_reg(self._func_regions, loc, nxt, prev_nodes)

   def inter_func_regional_connections(self, weighted: bool = True, distinct: bool = False) -> np.ndarray:
      """
      Returns how many connections each node has to different functional regions

      Parameters
      ----------
      weighted : bool, optional
          Whether or not to sum the weights of the streamlines of the inter-functional-regional connections, by default True
      distinct : bool, optional
          Whether to count the number of distinct inter-functional-regional connections or the total number of inter-functional-regional connections (only used if weighted=False), by default False

      Returns
      -------
      np.ndarray
          How many inter-functional-regional connections each node has
      """

      return self._int_reg_con(self._func_regions, weighted, distinct)

   # Measures not in use

   def edge_length(self) -> np.ndarray:
      """
      Returns the Euclidean distance between all pairs of adjacent nodes

      Returns
      -------
      np.ndarray
         Matrix of edge lengths
      """

      return self.euc_dist

   def edge_angle_change(self, loc: int, nxt: int, prev_nodes: List[int]) -> float:
      """
      Returns the magnitude of the deviation from the previous edges direction (radians)

      Parameters
      ----------
      loc : int
         Current node
      nxt : int
         Next node
      prev_nodes : List[int]
         Path sequence (excluding current node)

      Returns
      -------
      float
         Magnitude of deviation (radians)
      """

      if not prev_nodes or loc == nxt: return 0
      last_prev = prev_nodes[-1]
      a = self.euc_dist[last_prev, loc]
      b = self.euc_dist[loc, nxt]
      c = self.euc_dist[last_prev, nxt]
      cosc = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
      return pi - acos(cosc)

   def node_strength_dissimilarity(self, weighted: bool = True, method: str = 'tot') -> np.ndarray:
      """
      Returns the magnitude of the difference in the sum of edge weights of any pair of nodes

      Parameters
      ----------
      weighted : bool, optional
         Whether to use the weighted or unweighted SC matrix, by default True
      method : str, optional
         Used if weighted=True, by default 'tot'
         - 'in'  : in-degree strengths only
         - 'out' : out-degree strengths only
         - 'tot' : combined in and out degree strengths

      Returns
      -------
      np.ndarray
         Matrix of node strength dissimilarity
      """

      M = self.node_strength(weighted, method)
      return abs(M.reshape(-1,1) - M.T)

   def triangle_node_prevalence(self) -> np.ndarray:
      """
      Returns the number of SC-FC triangles centred (apex) on each node

      Returns
      -------
      np.ndarray
         Vector of SC-FC triangle node prevalances
      """

      T = (self.sc_bin @ (self.fc_bin * (1 - self.sc_bin.T)) @ self.sc_bin).diagonal()
      return T if self._sc_directed else T / 2

   def triangle_edge_prevalence(self) -> np.ndarray:
      """
      Returns the number of SC-FC triangles that involve each SC edge

      Returns
      -------
      np.ndarray
         Matrix of edge prevalances in SC-FC triangles
      """

      A = self.fc_bin * (1 - self.sc_bin)
      sc_T = self.sc_bin.T
      return self.sc_bin * (A @ sc_T + sc_T @ A)

   def hops_to_prev_used_nodes(self, nxt: int, prev_nodes: List[int]) -> int:
      """
      Returns the lowest number of hops from any previously visited nodes to the potential next node
      (i.e. shortest path from previous nodes to the potential next node)

      Parameters
      ----------
      nxt : int
         Next node
      prev_nodes : List[int]
         Path sequence (containing previously visited nodes)

      Returns
      -------
      int
         Fewest number of hops to a previously used node
      """

      if not prev_nodes:
         return 0
      return self.shortest_paths(method='hops')[:,nxt][prev_nodes].min()

   def dist_to_prev_used_nodes(self, nxt: int, prev_nodes: List[int]) -> float:
      """
      Returns the Euclidean distance of the potential next node to the closest previously visited node

      Parameters
      ----------
      nxt : int
         Next node
      prev_nodes : List[int]
         Path sequence (containing previously visited nodes)

      Returns
      -------
      float
         Euclidean distance to closest visited node
      """

      if not prev_nodes:
         return 0
      return self.euc_dist[nxt][prev_nodes].min()

   def shortest_paths(self, method: str = 'hops') -> np.ndarray:
      """
      Returns the shortest path lengths between any two nodes

      Parameters
      ----------
      method : str, optional
         Determines how the shortest paths are calculated, by default 'hops'
         - 'hops' : binary / fewest hops
         - 'dist' : sum of edge distances

      Returns
      -------
      np.ndarray
         Matrix of shortest path lengths
      """

      if method == 'hops':
         if self._sp_hops is None:
            self._sp_hops = dijkstra(self.sc_bin, directed=self.sc_directed, unweighted=True)
            self._sp_hops[self._sp_hops == np.inf] = -1
         return self._sp_hops
      elif method == 'dist':
         if self._sp_dist is None:
            self._sp_dist = dijkstra(self.sc_bin * self.euc_dist, directed=self.sc_directed, unweighted=False)
            self._sp_dist[self._sp_dist == np.inf] = -1
         return self._sp_dist
      else:
         raise ValueError("Invalid method")


   # Internal

   @staticmethod
   def _get_bin_matrix(M, M_thresh, M_thresh_type):
      """
      Returns a validated binary matrix with a zero diagonal
      M : weighted adjacency matrix
      M_thresh : threshold to apply
      M_thresh_type : threshold type to apply
      """

      M_bin = binarise_matrix(M, M_thresh, M_thresh_type)
      np.fill_diagonal(M_bin, 0)
      return M_bin

   @staticmethod
   def _get_wei_matrix(M):
      """
      Returns a validated weighted adjacency matrix with a zero diagonal
      M : weighted adjacency matrix
      """

      validate_square(M)
      M_wei = M.copy()
      np.fill_diagonal(M_wei, 0)
      return M_wei

   @staticmethod
   def _get_regions(M, res):
      """
      Raises an exception if the assigned regions are invalid
      M : vector of regions assigned to each node
      res : resolution of the brain
      """

      regions = np.array(M, dtype=int) if M is not None else np.array([], dtype=int)
      if len(regions) != 0:
         if len(regions) < res or len(regions.shape) > 1:
            raise ValueError("Invalid regions")
      return regions

   @staticmethod
   def _is_targ_reg(regions, nxt, target):
      """
      Returns whether or not a potential next node is in a defined target region
      regions : Defined regions for each node
      nxt : Next node
      target : Target node
      """

      return int(regions[nxt] == regions[target])

   @staticmethod
   def _edge_con_diff_reg(regions, loc, nxt, target):
      """
      Returns whether or not a potential next node leaves a predefined region, if it is not already in the target region
      regions : Defined regions for each node
      loc : Current node location
      nxt : Next node
      target : Target node
      """

      # Moving to target func region
      r_nxt = regions[nxt]
      r_target = regions[target]
      if r_nxt == r_target:
         return 1

      # Leaving target func region
      r_loc = regions[loc]
      if r_loc == r_target:
         return 0

      # Changing non-target func region
      return int(r_loc != r_nxt)

   @staticmethod
   def _prev_vis_reg(regions, loc, nxt, prev_nodes):
      """
      Returns whether or not a predefined region of a potential next node has already been visited, unless it remains in the same region
      regions : Defined regions for each node
      loc : Current node location
      nxt : Next node
      prev_nodes : Previously visited nodes
      """

      nxt_r = regions[nxt]
      if nxt_r == regions[loc]:
         return 0
      for p in prev_nodes:
         if nxt_r == regions[p]:
            return 1
      return 0

   def _int_reg_con(self, regions, weighted=True, distinct=False):
      """
      Returns how many connections each node has to different predefined regions

      Parameters
      ----------
      regions : Defined regions for each node
      weighted : Whether or not to sum the weights of the streamlines of the inter-regional connections
      distinct : Whether to count the number of distinct inter-regional connections or the total number of inter-regional connections (only used if weighted=False)
      """

      M = np.zeros(self.res)
      for i in range(self.res):
         r = regions[i]

         # Only include where an edge to a different region exists
         mask = np.where((self.sc_bin[i] > 0) & (regions != r))

         if weighted:
            M[i] = self.sc[i][mask].sum()
         elif not distinct:
            M[i] = self.sc_bin[i][mask].sum()
         else:
            M[i] = len(set(regions[mask]))

      return M