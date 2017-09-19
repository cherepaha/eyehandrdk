import pandas as pd
import numpy as np

class DerivativeCalculator:  
    def append_derivatives(self, dynamics):
        dynamics = dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no'], 
                                    group_keys=False).apply(self.get_velocity)
        return dynamics
        
    def get_velocity(self, trajectory):
        mouse_vx = self.differentiate(trajectory.timestamp.values, trajectory.mouse_x.values)
        mouse_vy = self.differentiate(trajectory.timestamp.values, trajectory.mouse_y.values)
        eye_vx = self.differentiate(trajectory.timestamp.values, trajectory.eye_x.values)
        eye_vy = self.differentiate(trajectory.timestamp.values, trajectory.eye_y.values)
                
        derivatives = pd.DataFrame(np.asarray([mouse_vx, mouse_vy, eye_vx, eye_vy]).T, 
                                   columns=['mouse_vx', 'mouse_vy', 'eye_vx', 'eye_vy'],
                                   index=trajectory.index)
        return pd.concat([trajectory, derivatives], axis=1)
    
#    use this if accelerations are also needed
    def get_derivatives(self, trajectory):
        mouse_vx, mouse_ax = self.differentiate(trajectory.timestamp.values, trajectory.mouse_x.values)
        mouse_vy, _ = self.differentiate(trajectory.timestamp.values, trajectory.mouse_y.values)
        eye_vx, eye_ax = self.differentiate(trajectory.timestamp.values, trajectory.eye_x.values)
        eye_vy, _ = self.differentiate(trajectory.timestamp.values, trajectory.eye_y.values)
                
        derivatives = pd.DataFrame(np.asarray([mouse_vx, mouse_vy, eye_vx, eye_vy, mouse_ax, eye_ax]).T, 
                                   columns=['mouse_vx', 'mouse_vy', 'eye_vx', 'eye_vy', 'mouse_ax', 'eye_ax'],
                                   index=trajectory.index)
        return pd.concat([trajectory, derivatives], axis=1)
        
    def differentiate(self, t, x):
        # TODO: currently, fixed timestep is assumed. Change the formula to allow for variable timestep
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