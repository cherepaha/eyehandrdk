from __future__ import division
import pandas as pd
import numpy as np
import derivative_calculator

class DataPreprocessor:
    x_lim = 1920
    y_lim = 1080
    n_steps = 50
    
    com_threshold_x = 50
    com_threshold_y = 100
    
    pupil_interp_window = 0.05
    
    def preprocess_data(self, choices, dynamics, flip_left=True):
        # these two trials have very poor eye data, so better drop them
#        choices.drop([(391, 1, 10, 59), (451, 1, 8, 27)], inplace=True)
#        dynamics.drop([(391, 1, 10, 59), (451, 1, 8, 27)], inplace=True)
        
        # originally, EyeLink data has -32768.0 values in place when data loss occurred
        # we replace it with np.nan to be able to use numpy functions properly
        dynamics = dynamics.replace(dynamics.eye_x.min(), np.nan)           
        
        dynamics = self.set_origin_to_start(dynamics)                   
        dynamics = self.shift_timeframe(dynamics)
        choices.response_time /= 1000.0
        
        if flip_left:
            dynamics = self.flip_left(choices, dynamics)
            
#        dynamics = self.resample_trajectories(dynamics)
        
        # append derivatives
        dc = derivative_calculator.DerivativeCalculator()
        dynamics = dc.append_derivatives(dynamics)
#        dynamics = dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']).\
#                                    apply(self.preprocess_pupil_size)
        
        # append various gaze and mouse measures
        choices['xflip_count'] = dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']).\
                                    apply(lambda traj: self.zero_cross_count(traj.mouse_vx))    
        choices = choices.join(dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']).\
                                    apply(self.get_maxd))
        choices = choices.join(dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']).\
                                    apply(self.get_midline_d))
        choices['is_com'] = ((choices.midline_d > self.com_threshold_x) & \
                                (choices.midline_d_y > self.com_threshold_y))
        choices = choices.join(dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']).\
                                    apply(self.get_initiation_time))
#        choices['desync_rate'] = dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']).\
#                                    apply(self.get_mouse_eye_desync_rate)
#        dynamics['id'] = choices['subj_id'].map(str)  + '_' + choices['session_no'].map(str) + \
#                        '_' + choices['block_no'].map(str) + '_' + choices['trial_no'].map(str)
                                            
#        analyser = data_analyser.DataAnalyser()        
#        choices = analyser.append_dwell_times(choices, dynamics)

        return choices, dynamics
    
    def preprocess_stim_viewing_data(self, choices, stim_viewing):
        stim_viewing = self.resample_trajectories(stim_viewing)
        stim_viewing = self.set_origin_to_start(stim_viewing)
        stim_viewing = self.flip_left(choices, stim_viewing)
                
        return stim_viewing
    
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
        dynamics.timestamp = dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']) \
                        ['timestamp'].apply(lambda t: (t-t.min()))/1000.0        
        
        return dynamics
        
    def flip_left(self, choices, dynamics):
        dynamics = dynamics.join(choices.direction)
        dynamics = dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']). \
                                    apply(self.reverse_x)
        dynamics.drop('direction', axis=1, inplace=True)
        return dynamics

    def resample_trajectories(self, dynamics):
        resampled_dynamics = dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']).\
                                    apply(self.resample_trajectory)
        resampled_dynamics.index = resampled_dynamics.index.droplevel(4)
        return resampled_dynamics
        
    def resample_trajectory(self, trajectory):
        # Make the sampling time intervals regular
        n = np.arange(0, self.n_steps+1)
        t_regular = np.linspace(trajectory.timestamp.min(), trajectory.timestamp.max(), self.n_steps+1)
        mouse_x_interp = np.interp(t_regular, trajectory.timestamp.values, trajectory.mouse_x.values)
        mouse_y_interp = np.interp(t_regular, trajectory.timestamp.values, trajectory.mouse_y.values)
        eye_x_interp = np.interp(t_regular, trajectory.timestamp.values, trajectory.eye_x.values)
        eye_y_interp = np.interp(t_regular, trajectory.timestamp.values, trajectory.eye_y.values)
        pupil_size_interp = np.interp(t_regular, trajectory.timestamp.values, 
                                      trajectory.pupil_size.values)
        traj_interp = pd.DataFrame([n, t_regular, mouse_x_interp, mouse_y_interp, \
                                    eye_x_interp, eye_y_interp, pupil_size_interp]).transpose()
        traj_interp.columns = ['n', 'timestamp', 'mouse_x', 'mouse_y', 'eye_x', 'eye_y', 'pupil_size']
#        traj_interp.index = range(1,self.n_steps+1)
        return traj_interp
    
    def reverse_x(self, trajectory):    
        # We need to reverse the trajectories for RDK direction = 180 so that correct 
        # response is always on the right-hand side
        # Here, we assume that the trajectory.x is already remapped to (-x_lim, x_lim)
        if (trajectory.iloc[0]['direction'] == 180):
            trajectory.loc[:,'mouse_x'] = -trajectory.loc[:,'mouse_x']
            trajectory.loc[:,'eye_x'] = -trajectory.loc[:,'eye_x']
        return trajectory

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

    def get_initiation_time(self, trajectory):
        trimmed_traj = trajectory.drop_duplicates(subset=['mouse_x', 'mouse_y'], keep='last')
        initiation_time = trimmed_traj.timestamp.min() - trajectory.timestamp.min()
        motion_time = trimmed_traj.timestamp.max() - trimmed_traj.timestamp.min()
        
        return pd.Series({'initiation_time': initiation_time, 
                          'motion_time': motion_time})
    
    def append_is_early_response(self, choices, dynamics):
        dynamics = dynamics.join(choices.initiation_time)
        choices['is_early_response'] = dynamics.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']). \
                        apply(lambda traj: traj.initiation_time.iloc[0]==0)
#                        apply(lambda traj: traj.mouse_vx.iloc[0]!=0)
        return choices
    
    def get_initiation_time_v1(self, trajectory):
        submovements = self.get_submovements(trajectory.vx.values)
        
        
    def get_submovements(self, vx):
        # TODO: look into old code for AP extraction from stick balancing data
        # in a simple for loop, go through 
        # return dataframe with columns idx_start, idx_end, t_start, t_end, distance
        pass
        
#    def preprocess_pupil_size(self, trajectory):
#        if (trajectory.pupil_size == 0).any():
#            ps = trajectory.pupil_size.values    
#            t = trajectory.timestamp.values
#            step = (t[1]-t[0])
#            n_steps_per_window = int(np.ceil(self.pupil_interp_window/step))
#            
#            zero_seq_endpoints = np.where(np.diff(ps == 0) != 0)[0]
#            
#            if ps[0]==0:
#                zero_seq_endpoints = np.concatenate(([0], zero_seq_endpoints))
#            if ps[-1]==0:
#                zero_seq_endpoints = np.concatenate((zero_seq_endpoints, [len(ps)-1]))
#            
#            for n_seq, first_zero_idx in enumerate(zero_seq_endpoints[::2]):
#                last_zero_idx = zero_seq_endpoints[2*n_seq+1]
#                ps[max(0, first_zero_idx+1-n_steps_per_window):first_zero_idx+1] = 0
#                ps[last_zero_idx+1: min(len(ps)-1, last_zero_idx+1+n_steps_per_window)] = 0
#    
#            ps_interp = np.interp(t, t[~(ps==0)], ps[~(ps==0)])
#                                  
#            trajectory['pupil_size'] = ps_interp
#        
#        trajectory['pupil_size_rel'] = trajectory['pupil_size'] /trajectory['pupil_size'].ix[0]-1
#        
#        return trajectory