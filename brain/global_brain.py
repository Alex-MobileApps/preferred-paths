from brain.brain import Brain
import numpy as np

class GlobalBrain(Brain):


   # Constructor

   def __init__(self, sc, fc, euc_dist, sc_directed=Brain._DEF_SC_DIR, sc_thresh=Brain._DEF_SC_THRESH, fc_thresh=Brain._DEF_FC_THRESH, sc_thresh_type=Brain._DEF_THRESH_TYPE, fc_thresh_type=Brain._DEF_THRESH_TYPE, hubs=None, regions=None):
      super().__init__(sc=sc, fc=fc, euc_dist=euc_dist, sc_directed=sc_directed, sc_thresh=sc_thresh, fc_thresh=fc_thresh, sc_thresh_type=sc_thresh_type, fc_thresh_type=fc_thresh_type, hubs=hubs, regions=regions)


   # Measures

   def closest_to_target(self):
      """
      Returns how much closer a target node becomes when moving to a node adjacent to the current location

      Returns
      -------
      out : numpy.ndarray
          How much closer a target node becomes for any target and edge.
          A 3D matrix indexed by out[target, loc, nxt]
      """

      n = self.res
      range_n = range(n)
      ctt = np.ones((n,n,n)) * np.inf
      for loc in range_n:
         for nxt in range_n:
            if self.sc_bin[loc, nxt] == 0:
               continue
            for target in range_n:
               ctt[target, loc, nxt] = self.euc_dist[loc, target] - self.euc_dist[nxt, target]
      return ctt