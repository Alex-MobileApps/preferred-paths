from brain.brain import Brain
import numpy as np

class GlobalBrain(Brain):


   # Constructor

   def __init__(self, sc, fc, euc_dist, sc_thresh, fc_thresh, sc_thresh_type=Brain._DEF_THRESH_TYPE, fc_thresh_type=Brain._DEF_THRESH_TYPE):
      super().__init__(sc, fc, euc_dist, sc_thresh, fc_thresh, sc_thresh_type, fc_thresh_type)


   # Measures

   def closest_to_target(self):
      n = self.res
      range_n = range(n)
      ctt = np.ones((n,n,n)) * np.inf
      for i in range_n:
         for j in range_n:
            if self.sc_bin[i, j] == 0:
               continue
            for target in range_n:
               ctt[target, i, j] = self.euc_dist[i, target] - self.euc_dist[j, target]
      return ctt