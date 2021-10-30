from brain.brain import Brain
import numpy as np

class GlobalBrain(Brain):


   # Constructor

   def __init__(self, sc, fc, euc_dist, sc_directed=Brain._DEF_SC_DIR, sc_thresh=Brain._DEF_SC_THRESH, fc_thresh=Brain._DEF_FC_THRESH, sc_thresh_type=Brain._DEF_THRESH_TYPE, fc_thresh_type=Brain._DEF_THRESH_TYPE, hubs=None, regions=None, func_regions=None):
      super().__init__(sc=sc, fc=fc, euc_dist=euc_dist, sc_directed=sc_directed, sc_thresh=sc_thresh, fc_thresh=fc_thresh, sc_thresh_type=sc_thresh_type, fc_thresh_type=fc_thresh_type, hubs=hubs, regions=regions, func_regions=func_regions)


   # Measures

   def closest_to_target(self, loc, nxt, target):
      """
      Returns how much closer a target node becomes when moving to a node adjacent to the current location

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
      out : numpy.ndarray
          How much closer a target node becomes for any target and edge.
          A 3D matrix indexed by out[target, loc, nxt]
      """

      if not self.sc_bin[loc,nxt]:
         return np.inf
      return self.euc_dist[loc, target] - self.euc_dist[nxt, target]