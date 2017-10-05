from __future__ import division
import pandas as pd
import numpy as np
import derivative_calculator

class DataPreprocessor:
    x_lim = 1920
    y_lim = 1080
    
    com_threshold_x = 50
    com_threshold_y = 100
    
    # to determine exact response intiation,
    # threshold for distance travelled by mouse cursor (in pixels) during single movement
    # it is needed when calculating initation time
    it_distance_threshold = 100
    eye_v_threshold = 200

    # these two trials have very poor eye data, so we exclude them
    excluded_trials = [(391, 1, 10, 59), (451, 1, 8, 27)]
        
    index = ['subj_id', 'session_no', 'block_no', 'trial_no']
    
    def preprocess_data(self, choices, dynamics, resample=0):
#        choices, dynamics = self.drop_excluded_trials(choices, dynamics, self.excluded_trials)
        
        # originally, EyeLink data has -32768.0 values in place when data loss occurred
        # we replace it with np.nan to be able to use numpy functions properly
        dynamics = dynamics.replace(dynamics.eye_x.min(), np.nan)
        
        dynamics = self.set_origin_to_start(dynamics)                   
        dynamics = self.shift_timeframe(dynamics)
        
        dynamics = self.flip_left(choices, dynamics)
        
        if resample:
            dynamics = self.resample_trajectories(dynamics, n=resample)
        
        dc = derivative_calculator.DerivativeCalculator()
        dynamics = dc.append_diff(dynamics)
        dynamics = dc.append_derivatives(dynamics)
        
        dynamics['mouse_v'] = np.sqrt(dynamics.mouse_vx**2 + dynamics.mouse_vy**2 )
        dynamics['eye_v'] = np.sqrt(dynamics.eye_vx**2 + dynamics.eye_vy**2 )        
                          
        return dynamics

    def drop_excluded_trials(self, choices, dynamics, trials):
        choices = choices.drop(trials, errors='ignore')
        
        # NB: this is a hack to deal with (supposedly) pandas bug.
        # When multiindex values are not unique (as in dynamics dataframes), 
        # drop doesn't really work, so we have to remove trials one-by-one
        dynamics = dynamics.reset_index()
        for trial in trials:
            dynamics = dynamics[~((dynamics.subj_id==trial[0]) & (dynamics.session_no==trial[1]) & 
                                  (dynamics.block_no==trial[2]) & (dynamics.trial_no==trial[3]))]
        dynamics = dynamics.set_index(self.index, drop=True)
        
        return choices, dynamics        
    
    def get_mouse_and_gaze_measures(self, choices, dynamics, stim_viewing):
        choices['is_correct'] = choices['direction'] == choices['response']
        choices.response_time /= 1000.0
        choices['xflips'] = dynamics.groupby(level=self.index).\
                                    apply(lambda traj: self.zero_cross_count(traj.mouse_vx))    
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_maxd))
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_midline_d))
        choices['is_com'] = ((choices.midline_d > self.com_threshold_x) & \
                                (choices.midline_d_y > self.com_threshold_y))
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_mouse_IT))
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_eye_IT))
        
        choices['early_it'] = stim_viewing.groupby(level=self.index).apply(
                lambda traj: traj.timestamp.max()-traj.timestamp[traj.mouse_dx==0].iloc[-1])
        
        return choices
    
    def set_origin_to_start(self, dynamics):
        # set origin to start button location
        dynamics.mouse_x -= self.x_lim/2
        dynamics.mouse_y = self.y_lim - dynamics.mouse_y
        dynamics.eye_x -= self.x_lim/2
        dynamics.eye_y = self.y_lim - dynamics.eye_y
        return dynamics
    
    def shift_timeframe(self, dynamics):
        # shift time to the timeframe beginning at 0 for each trajectory
        # also, express time in seconds rather than milliseconds
        dynamics.loc[:,'timestamp'] = dynamics.timestamp.groupby(by=self.index). \
                                        transform(lambda t: (t-t.min()))/1000.0
        return dynamics

    def flip_left(self, choices, dynamics):
        for col in ['mouse_x', 'eye_x']:
            dynamics.loc[choices.direction==180, ['mouse_x', 'eye_x']] *= -1
        return dynamics

    def resample_trajectories(self, dynamics, n_steps=100):
        resampled_dynamics = dynamics.groupby(level=self.index).\
                                    apply(lambda traj: self.resample_trajectory(traj, n_steps=n_steps))
        resampled_dynamics.index = resampled_dynamics.index.droplevel(4)
        return resampled_dynamics
        
    def resample_trajectory(self, trajectory, n_steps):
        # Make the sampling time intervals regular
        n = np.arange(0, n_steps+1)
        t_regular = np.linspace(trajectory.timestamp.min(), trajectory.timestamp.max(), n_steps+1)
        mouse_x_interp = np.interp(t_regular, trajectory.timestamp.values, trajectory.mouse_x.values)
        mouse_y_interp = np.interp(t_regular, trajectory.timestamp.values, trajectory.mouse_y.values)
        eye_x_interp = np.interp(t_regular, trajectory.timestamp.values, trajectory.eye_x.values)
        eye_y_interp = np.interp(t_regular, trajectory.timestamp.values, trajectory.eye_y.values)
        pupil_size_interp = np.interp(t_regular, trajectory.timestamp.values, 
                                      trajectory.pupil_size.values)
        traj_interp = pd.DataFrame([n, t_regular, mouse_x_interp, mouse_y_interp, \
                                    eye_x_interp, eye_y_interp, pupil_size_interp]).transpose()
        traj_interp.columns = ['n', 'timestamp', 'mouse_x', 'mouse_y', 'eye_x', 'eye_y', 'pupil_size']
#        traj_interp.index = range(1,n_steps+1)
        return traj_interp
    
    def get_maxd(self, trajectory):
        alpha = np.arctan((trajectory.mouse_y.iloc[-1]-trajectory.mouse_y.iloc[0])/ \
                            (trajectory.mouse_x.iloc[-1]-trajectory.mouse_x.iloc[0]))
        d = (trajectory.mouse_x.values-trajectory.mouse_x.values[0])*np.sin(-alpha) + \
            (trajectory.mouse_y.values-trajectory.mouse_y.values[0])*np.cos(-alpha)
        if abs(d.min())>abs(d.max()):
            return pd.Series({'max_d': d.min(), 'idx_max_d': d.argmin()})
        else:
            return pd.Series({'max_d': d.max(), 'idx_max_d': d.argmax()})
        
    def get_midline_d(self, trajectory):
        mouse_x = trajectory.mouse_x.values
        is_final_point_positive = (mouse_x[-1]>0)
        
#        if is_final_point_positive:
#            midline_d = abs(mouse_x.min())
##            idx_midline_d = int(mouse_x.argmin())
#        else:
#            midline_d = abs(mouse_x.max())
##            idx_midline_d = int(mouse_x.argmax())
        midline_d = mouse_x.min() if is_final_point_positive else mouse_x.max()
#        print(trajectory)
#        print(midline_d)
#        print(abs(mouse_x.argmin()) if is_final_point_positive else abs(mouse_x.argmax()))
#        print((mouse_x == midline_d).nonzero())
        idx_midline_d = (mouse_x == midline_d).nonzero()[0][-1]
        midline_d_y = trajectory.mouse_y.values[idx_midline_d]
        return pd.Series({'midline_d': abs(midline_d), 
                          'idx_midline_d': idx_midline_d,
                          'midline_d_y': midline_d_y})

    def zero_cross_count(self, x):
        return (abs(np.diff(np.sign(x))) > 1).sum()

    def get_mouse_IT(self, traj):
        v = traj.mouse_v.values
    
        onsets = []
        offsets = []
        is_previous_v_zero = True
    
        for i in np.arange(0,len(v)):
            if v[i]!=0:
                if is_previous_v_zero:
                    is_previous_v_zero = False
                    onsets += [i]
                elif (i==len(v)-1):
                    offsets += [i]            
            elif (not is_previous_v_zero):
                offsets += [i]
                is_previous_v_zero = True
    
        submovements = pd.DataFrame([{'on': onsets[i], 
                 'off': offsets[i], 
                 'on_t': traj.timestamp.values[onsets[i]],
                 'distance':(traj.mouse_v[onsets[i]:offsets[i]]*
                             traj.timestamp.diff()[onsets[i]:offsets[i]]).sum()}
                for i in range(len(onsets))])
    
        it = submovements.loc[submovements.distance.ge(self.it_distance_threshold ).idxmax()].on_t
        return pd.Series({'mouse_IT': it, 'motion_time': traj.timestamp.max()-it})
    
    def get_eye_IT(self, trajectory):        
        eye_v = trajectory.eye_v
        if ((eye_v < self.eye_v_threshold) | (eye_v.isnull())).all():
            # if eye stays at one location throughout the whole trial, 
            # it is supposedly fixated either at the center of the screen, or at the response location
            # in the former case, initation time is inf, in the latter case, initation time is 0
            # we detect which case is tru  by looking at first value of eye_x
            if abs(trajectory.eye_x.iloc[0] < 100): 
                eye_IT = np.inf
                eye_initial_decision = 0
            else:
                eye_IT = 0
                eye_initial_decision = np.sign(trajectory.eye_x.iloc[0])
        else:
            eye_IT_idx = (eye_v > self.eye_v_threshold).nonzero()[0][0]
            eye_IT = trajectory.timestamp.iloc[eye_IT_idx]
            eye_initial_decision = np.sign(trajectory.eye_x.iloc[eye_IT_idx+1])
        
        return pd.Series({'eye_IT': eye_IT, 'eye_initial_decision': eye_initial_decision})

    def get_early_initiation_time(self, trajectory):        
        return trajectory.timestamp.max() - trajectory.timestamp[trajectory.mouse_dx==0].iloc[-1]        
        
    def append_is_early_response(self, choices, dynamics):
        dynamics = dynamics.join(choices.initiation_time)
        choices['is_early_response'] = dynamics.groupby(level=self.index). \
                        apply(lambda traj: traj.initiation_time.iloc[0]==0)
        return choices