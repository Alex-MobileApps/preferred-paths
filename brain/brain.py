import numpy as np
from utils import binarise_matrix, validate_square, validate_thresh_type, validate_symmetric
from math import pi, acos

class Brain():

   # Static variables

   _DEF_THRESH_TYPE = 'pos'


   # Constructor

   def __init__(self, sc, fc, euc_dist, sc_thresh, fc_thresh, sc_thresh_type=_DEF_THRESH_TYPE, fc_thresh_type=_DEF_THRESH_TYPE):
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
      sc_thresh, fc_thresh : float
         Threshold used to binarise SC and FC
      sc_thresh_type, fc_thresh_type : str
         Method used to binarise SC and FC matrices
         - 'pos' : values less than thresh_val are given 0. Other values are given 1.
         - 'neg' : values greater than thresh_val are given 0. Other values are given 1.
         - 'abs' : absolute values less than the absolute value of thresh_val are given 0. Other values are given 1.
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

   def edge_angle_change(self, source, target, prev):
      """
      Returns the magnitude of the deviation from the previous edges direction (radians)

      Parameters
      ----------
      source : int
         Current path location
      target : int
         Next node in the path
      prev : list
         Path sequence (excluding source node)

      Returns
      -------
      out : float
         Magnitude of deviation (radians)
      """

      if not prev or source == target: return 0
      last_prev = prev[-1]
      a = self.euc_dist[last_prev, source]
      b = self.euc_dist[source, target]
      c = self.euc_dist[last_prev, target]
      cosc = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
      return pi - acos(cosc)

   def node_strength(self, weighted=True):
      """
      Returns the sum of edge weights adjacent to a node

      Parameters
      ----------
      weighted : bool
         Whether to use the weighted or unweighted SC matrix

      Returns
      -------
      out : numpy.ndarray
         Vector of strengths for each node
      """

      M = self.sc if weighted else self.sc_bin
      return M.sum(axis=1)

   def node_strength_disimilarity(self, weighted=True):
      """
      Returns the magnitude of the difference in the sum of edge weights of any pair of nodes

      Parameters
      ----------
      weighted : bool
         Whether to use the weighted or unweighted SC matrix

      Returns
      -------
      out : numpy.ndarray
         Matrix of node strength disimilarity
      """

      M = self.node_strength(weighted)
      return abs(M.reshape(-1,1) - M.T)

   def triangle_node_prevalence(self):
      """
      Returns the number of SC-FC triangles centred (apex) on each node

      Returns
      -------
      out : numpy.ndarray
         Vector of SC-FC triangle node prevalances
      """

      return (self.sc_bin @ (self.fc_bin * (1 - self.sc_bin)) @ self.sc_bin).diagonal() / 2

   def triangle_edge_prevalence(self):
      """
      Returns the number of SC-FC triangles that involve each SC edge

      Returns
      -------
      out : numpy.ndarray
         Matrix of edge prevelances in SC-FC triangles
      """

      A = self.fc_bin * (1 - self.sc_bin)
      return self.sc_bin * (A @ self.sc_bin + self.sc_bin @ A)

   def dist_to_prev_used(self, target, prev):
      """
      Returns the Euclidean distance of the target node to the closest previously visited node

      Parameters
      ----------
      target : int
         Target node
      prev : list
         Path sequence (containing previously visited nodes)

      Returns
      -------
      out : float
         Euclidean distance to closest visited node
      """

      if not prev:
         return 0
      return self.euc_dist[target][prev].min()


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