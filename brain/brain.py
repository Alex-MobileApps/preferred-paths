import numpy as np
from utils import binarise_matrix, validate_square, validate_thresh_type
from math import pi, acos
from scipy.sparse.csgraph import dijkstra

class Brain():

   # Static variables

   _DEF_THRESH_TYPE = 'pos'
   _DEF_SC_THRESH = 1
   _DEF_FC_THRESH = 0.01
   _DEF_SC_DIR = False


   # Constructor

   def __init__(self, sc, fc, euc_dist, sc_directed=_DEF_SC_DIR, sc_thresh=_DEF_SC_THRESH, fc_thresh=_DEF_FC_THRESH, sc_thresh_type=_DEF_THRESH_TYPE, fc_thresh_type=_DEF_THRESH_TYPE, hubs=None, regions=None):
      """
      Contains weighted and binarised connectome matrices of a brain, as well as functions that compute features of these connectomes

      Parameters
      ----------
      sc : numpy.ndarray
         Weighted structural connectivity matrix
      fc : numpy.ndarray
         Weighted functional connectivity matrix
      euc_dist : numpy.ndarray
         Euclidean distances between each node pair
      sc_directed : bool
         Whether or not the SC layer is a directed graph or not
      sc_thresh, fc_thresh : float
         Threshold used to binarise SC and FC
      sc_thresh_type, fc_thresh_type : str
         Method used to binarise SC and FC matrices
         - 'pos' : values less than thresh_val are given 0. Other values are given 1.
         - 'neg' : values greater than thresh_val are given 0. Other values are given 1.
         - 'abs' : absolute values less than the absolute value of thresh_val are given 0. Other values are given 1.
      hubs : numpy.ndarray
         Hub node indexes
      regions : numpy.ndarray
         Region (as an integer) that each node is assigned to
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

      self._hubs = np.array(hubs, dtype=np.int) if hubs is not None else np.array([], dtype=np.int)
      if len(self._hubs) != 0:
         if self._hubs.min() < 0 or self._hubs.max() > len(self._sc) - 1 or len(self._hubs.shape) > 1:
            raise ValueError("Invalid hub node indexes")

      self._regions = np.array(regions, dtype=np.int) if regions is not None else np.array([], dtype=np.int)
      if len(self._regions) != 0:
         if len(self._regions) < self.res or len(self._regions.shape) > 1:
            raise ValueError("Invalid regions")


   # Properties

   @property
   def sc(self):
      """
      Get the weighted SC matrix
      """
      return self._sc

   @property
   def sc_bin(self):
      """
      Get the binarised SC matrix
      """
      return self._sc_bin

   @property
   def sc_thresh(self):
      """
      Get the threshold used to create the binarised SC matrix
      """
      return self._sc_thresh

   @property
   def sc_thresh_type(self):
      """
      Get the threshold type used to create the binarised SC matrix
      """
      return self._sc_thresh_type

   @property
   def sc_directed(self):
      """
      Get whether or not the SC layer is a directed graph or not
      """
      return self._sc_directed

   @property
   def fc(self):
      """
      Get the weighted FC matrix
      """
      return self._fc

   @property
   def fc_bin(self):
      """
      Get the binarised FC matrix
      """
      return self._fc_bin

   @property
   def fc_thresh(self):
      """
      Get the threshold used to create the binarised FC matrix
      """
      return self._fc_thresh

   @property
   def fc_thresh_type(self):
      """
      Get the threshold type used to create the binarised FC matrix
      """
      return self._fc_thresh_type

   @property
   def euc_dist(self):
      """
      Get the Euclidean distances matrix
      """
      return self._euc_dist

   @property
   def res(self):
      """
      Get the parcellation resolution
      """
      return len(self._sc)


   # Measures

   def streamlines(self, weighted=True):
      """
      Returns the number of streamlines between any pair of nodes

      Parameters
      ----------
      weighted : bool
         Whether to use the weighted or unweighted SC matrix

      Returns
      -------
      out : numpy.ndarray
         Matrix of streamlines
      """

      M = self.sc if weighted else self.sc_bin
      return M

   def edge_length(self):
      """
      Returns the Euclidean distance between all pairs of adjacent nodes

      Returns
      -------
      out : numpy.ndarray
         Matrix of edge lengths
      """

      return self.euc_dist

   def edge_angle_change(self, loc, nxt, prev_nodes):
      """
      Returns the magnitude of the deviation from the previous edges direction (radians)

      Parameters
      ----------
      loc : int
         Current node
      nxt : int
         Next node
      prev_nodes : list
         Path sequence (excluding current node)

      Returns
      -------
      out : float
         Magnitude of deviation (radians)
      """

      if not prev_nodes or loc == nxt: return 0
      last_prev = prev_nodes[-1]
      a = self.euc_dist[last_prev, loc]
      b = self.euc_dist[loc, nxt]
      c = self.euc_dist[last_prev, nxt]
      cosc = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
      return pi - acos(cosc)

   def node_strength(self, weighted=True, method='tot'):
      """
      Returns the sum of edge weights adjacent to a node

      Parameters
      ----------
      weighted : bool
         Whether to use the weighted or unweighted SC matrix
      method : str
         Used if weighted=True
         - 'in'  : in-degree strengths only
         - 'out' : out-degree strengths only
         - 'tot' : combined in and out degree strengths

      Returns
      -------
      out : numpy.ndarray
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

   def node_strength_dissimilarity(self, weighted=True, method='tot'):
      """
      Returns the magnitude of the difference in the sum of edge weights of any pair of nodes

      Parameters
      ----------
      weighted : bool
         Whether to use the weighted or unweighted SC matrix
      method : str
         Used if weighted=True
         - 'in'  : in-degree strengths only
         - 'out' : out-degree strengths only
         - 'tot' : combined in and out degree strengths

      Returns
      -------
      out : numpy.ndarray
         Matrix of node strength dissimilarity
      """

      M = self.node_strength(weighted, method)
      return abs(M.reshape(-1,1) - M.T)

   def triangle_node_prevalence(self):
      """
      Returns the number of SC-FC triangles centred (apex) on each node

      Returns
      -------
      out : numpy.ndarray
         Vector of SC-FC triangle node prevalances
      """

      T = (self.sc_bin @ (self.fc_bin * (1 - self.sc_bin.T)) @ self.sc_bin).diagonal()
      return T if self._sc_directed else T / 2

   def triangle_edge_prevalence(self):
      """
      Returns the number of SC-FC triangles that involve each SC edge

      Returns
      -------
      out : numpy.ndarray
         Matrix of edge prevalances in SC-FC triangles
      """

      A = self.fc_bin * (1 - self.sc_bin)
      sc_T = self.sc_bin.T
      return self.sc_bin * (A @ sc_T + sc_T @ A)

   def hops_to_prev_used_nodes(self, nxt, prev_nodes):
      """
      Returns the lowest number of hops from any previously visited nodes to the potential next node
      (i.e. shortest path from previous nodes to the potential next node)

      Parameters
      ----------
      nxt : int
         Next node
      prev_nodes : list
         Path sequence (containing previously visited nodes)

      Returns
      -------
      out : int
         Fewest number of hops to a previously used node
      """

      if not prev_nodes:
         return 0
      return self.shortest_paths(method='hops')[:,nxt][prev_nodes].min()

   def dist_to_prev_used_nodes(self, nxt, prev_nodes):
      """
      Returns the Euclidean distance of the potential next node to the closest previously visited node

      Parameters
      ----------
      nxt : int
         Next node
      prev_nodes : list
         Path sequence (containing previously visited nodes)

      Returns
      -------
      out : float
         Euclidean distance to closest visited node
      """

      if not prev_nodes:
         return 0
      return self.euc_dist[nxt][prev_nodes].min()

   def is_target_node(self, nxt, target):
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
      out : int
         Whether or not the potential next node is the target node (1 if true, 0 otherwise)
      """

      return int(nxt == target)

   def is_target_region(self, nxt, target):
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
      out : int
          Whether or not the potential next node is the target node's region (1 if true, 0 otherwise)
      """

      return int(self._regions[nxt] == self._regions[target])

   def shortest_paths(self, method='hops'):
      """
      Returns the shortest path lengths between any two nodes

      Parameters
      ----------
      method : str
         Determines how the shortest paths are calculated
         - 'hops' : binary / fewest hops
         - 'dist' : sum of edge distances

      Returns
      -------
      out : numpy.ndarray
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

   def hubs(self, binary=False):
      """
      Returns the brain's hub nodes

      Parameters
      ----------
      binary : bool
         Whether to return the indexes of the hub nodes, or a binary array for all nodes indicating which ones are hubs

      Returns
      -------
      out : numpy.ndarray
         Binary vector indicating which nodes are hub nodes
      """

      if not binary:
         return self._hubs
      else:
         M = np.zeros(self.res, dtype=np.int)
         M[self._hubs] = 1
         return M

   def neighbour_just_visited_node(self, nxt, prev_nodes):
      """
      Returns whether or not a potential next node neighbours the most recently visited node

      Parameters
      ----------
      nxt : int
          Next node
      prev_nodes : list
          Path sequence (containing previously visited nodes)

      Returns
      -------
      out : int
          Whether or not the potential next node neighbours the just visited node (1 if true, 0 otherwise)
      """

      if not prev_nodes:
         return 0
      return self.sc_bin[prev_nodes[-1], nxt]

   def leave_non_target_region(self, loc, nxt, target):
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
      out : int
          1 if leaving a non-target region or remaining in the target region, 0 otherwise
      """

      # Moving to target region
      r_nxt = self._regions[nxt]
      r_target = self._regions[target]
      if r_nxt == r_target:
         return 1

      # Leaving target region
      r_loc = self._regions[loc]
      if r_loc == r_target:
         return 0

      # Changing non-target region
      return int(r_loc != r_nxt)


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