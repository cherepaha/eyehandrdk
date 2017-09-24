import pandas as pd
import numpy as np

class DerivativeCalculator:   
    def append_diff(self, dynamics):
        index = ['subj_id', 'session_no', 'block_no', 'trial_no']
        names = {'mouse_x': 'mouse_dx', 
                 'mouse_y': 'mouse_dy' , 
                 'eye_x': 'eye_dx', 
                 'eye_y': 'eye_dy'}      

        for col_name, der_name in names.items():
            dynamics[der_name] = np.concatenate(
                    [self.get_diff(traj['timestamp'].values, traj[col_name].values) 
                            for traj_id, traj in dynamics.groupby(level=index, group_keys=False)]
                    )
        return dynamics
    
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
    
    def get_diff(self, t, x):
        return np.concatenate(([0.], np.diff(x)/np.diff(t)))
    
    def differentiate(self, t, x):
        # To be able to reasonably calculate derivatives at the end-points of the trajectories,
        # I append three extra points before and after the actual trajectory, so we get N+6
        # points instead of N       
        x = np.append(x[0]*np.ones(3), np.append(x, x[-1]*np.ones(3)))
        
        # Time vector is also artificially extended by equally spaced points
        # Use median timestep to add dummy points to the time vector
        timestep = np.median(np.diff(t))
        t = np.append(t[0]-np.arange(1,4)*timestep, np.append(t, t[-1]+np.arange(1,4)*timestep))

        # smooth noise-robust differentiators, see: 
        # http://www.holoborodko.com/pavel/numerical-methods/ \
        # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
        v = (1*(x[6:]-x[:-6])/((t[6:]-t[:-6])/6) + 
             4*(x[5:-1] - x[1:-5])/((t[5:-1]-t[1:-5])/4) + 
             5*(x[4:-2] - x[2:-4])/((t[4:-2]-t[2:-4])/2))/32

        return v