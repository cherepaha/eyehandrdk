import pandas as pd
import numpy as np

class DerivativeCalculator:  
    def append_derivatives(self, dynamics):
        index = ['subj_id', 'session_no', 'block_no', 'trial_no']
        names = {'mouse_x': 'mouse_vx', 
                 'mouse_y': 'mouse_vy' , 
                 'eye_x': 'eye_vx', 
                 'eye_y': 'eye_vy'}      

        for col_name, der_name in names.items():
            dynamics[der_name] = np.concatenate(
                    [self.differentiate(traj['timestamp'].values, traj[col_name].values) 
                            for traj_id, traj in dynamics.groupby(level=index, group_keys=False)]
                    )
        return dynamics
           
    def differentiate(self, t, x):
        step = (t[1]-t[0])

        # To be able to reasonably calculate derivatives at the end-points of the trajectories,
        # I append three extra points before and after the actual trajectory, so we get N+6
        # points instead of N       
        x = np.append(x[0]*np.ones(3),np.append(x, x[-1]*np.ones(3)))

        # smooth noise-robust differentiators: http://www.holoborodko.com/pavel/numerical-methods/ \
        # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
        v = (-x[:-6] - 4*x[1:-5] - 5*x[2:-4] + 5*x[4:-2] + 4*x[5:-1] + x[6:])/(32*step)
#        a = (x[:-6] + 2*x[1:-5] - x[2:-4] - 4*x[3:-3] - x[4:-2] + 2*x[5:-1]+x[6:])\
#                /(16*step*step)
        return v