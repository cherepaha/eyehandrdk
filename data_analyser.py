#from __future__ import division
import pandas as pd
import numpy as np
import seaborn as sns
import random

class DataAnalyser:
    x_lim = 1920
    y_lim = 1080
    resp_area_radius = 175
    resp_area_offset = 25
    center_AOI_radius = 75
    
    def get_random_trajectory(self, choices, dynamics, n=1):
        return dynamics.loc[random.sample(list(choices.index), n)]

    # This one is wrong!
    def get_psychometric_function(self, choices, variables = ['subj_id', 'coherence'], 
                                  flatten=True):
        n_correct = choices[choices.is_correct==True].groupby(variables).size()
        n = choices.groupby(variables).size()
            
        rate_correct = n_correct/n
        rate_correct.name = 'p_correct'
        
        if flatten:
            rate_correct = rate_correct.to_frame().reset_index()
            rate_correct.fillna(0)
        return rate_correct
    
    def get_IT(self, choices, variables = ['subj_id', 'coherence'], norm=False):
        variable = 'initiation_time_norm' if norm else 'initiation_time'
        rt = choices.groupby(variables).mean()[variable]
        rt.name = variable
#        count = choices.groupby(variables).size()
#        rt.name = 'count'
        
        rt = rt.to_frame().reset_index()
#        rt['count'] = choices.groupby(variables).size()        
        return rt       
    
    def get_p_right(self, choices, variables = ['subj_id', 'coherence'], blocks=None):
        n_right = choices[choices.response==0]. \
            groupby(variables).agg('count').response
        n = choices.groupby(variables).agg('count').response
            
        p_right = n_right/n
        p_right.name = 'p_right'
        p_right = p_right.to_frame().fillna(0).reset_index()
        return p_right
    
    def append_dwell_times(self, choices, dynamics):
        incorrect_resp_area_positions, correct_resp_area_positions = \
                                            self.get_resp_area_positions(dynamics)
        dynamics_temp = dynamics.join(incorrect_resp_area_positions, rsuffix='_incorrect_resp_area_center')
        dynamics_temp = dynamics_temp.join(correct_resp_area_positions, 
                                           rsuffix='_correct_resp_area_center')
        dwell_times = dynamics_temp.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']).\
                                    apply(self.get_dwell_times)
        choices = choices.join(dwell_times)
        choices['dwell_chosen'] = choices.dwell_incorrect*(1-choices.is_correct) + \
                            choices.dwell_correct*choices.is_correct
        choices['dwell_chosen_not_cursor'] = choices.dwell_incorrect_not_cursor*(1-choices.is_correct) + \
                            choices.dwell_correct_not_cursor*choices.is_correct
        # if is_correct = False, dwell_unchosen is set to dwell_correct
        choices['dwell_unchosen'] = choices.dwell_incorrect*choices.is_correct + \
                                    choices.dwell_correct*(1-choices.is_correct)
        choices['dwell_unchosen_not_cursor'] = choices.dwell_incorrect_not_cursor*choices.is_correct + \
                                    choices.dwell_correct_not_cursor*(1-choices.is_correct)
#        choices['dwell_cursor'] = choices['dwell_cursor']                            
        
#        choices['dwell_chosen'] = choices['dwell_chosen_norm']*choices['response_time']
#        choices['dwell_unchosen'] = choices['dwell_unchosen_norm']*choices['response_time']
#        choices['dwell_cursor'] = choices['dwell_cursor_norm']*choices['response_time']
        
        return choices
    
    def get_resp_area_positions(self, dynamics, offset = 1080*0.75):
        incorrect_resp_area_positions = dynamics[(dynamics.eye_x<0) & (dynamics.eye_y>offset)]. \
                                    loc[:,['eye_x', 'eye_y']].groupby(level='subj_id').mean()
        correct_resp_area_positions = dynamics[(dynamics.eye_x>0) & (dynamics.eye_y>offset)]. \
                                    loc[:,['eye_x', 'eye_y']].groupby(level='subj_id').mean()        
        return incorrect_resp_area_positions, correct_resp_area_positions
        
    def get_dwell_times(self, trajectory):
#        incorrect_resp_area_center = [-(960-150), (1080-150)]
#        correct_resp_area_center = [(960-150), (1080-150)]
        mouse_cursor_radius = 100
#        dynamics = self.get_resp_area_positions(dynamics)
        
        is_in_cursor_area = (trajectory.eye_x - trajectory.mouse_x)**2 + \
                    (trajectory.eye_y - trajectory.mouse_y)**2 < mouse_cursor_radius**2
        is_in_incorrect_resp_area = (trajectory.eye_x - trajectory.eye_x_incorrect_resp_area_center)**2 + \
                    (trajectory.eye_y - trajectory.eye_y_incorrect_resp_area_center)**2 < self.resp_area_radius**2
        is_in_correct_resp_area = (trajectory.eye_x - trajectory.eye_x_correct_resp_area_center)**2 + \
                    (trajectory.eye_y - trajectory.eye_y_correct_resp_area_center)**2 < self.resp_area_radius**2
        
        is_in_incorrect_resp_area_not_cursor = (is_in_incorrect_resp_area) & \
                    ((trajectory.mouse_x - trajectory.eye_x_incorrect_resp_area_center)**2 + \
                    (trajectory.mouse_y - trajectory.eye_y_incorrect_resp_area_center)**2 < self.resp_area_radius**2)

        is_in_correct_resp_area_not_cursor = (is_in_correct_resp_area) & \
                    ((trajectory.mouse_x - trajectory.eye_x_correct_resp_area_center)**2 + \
                    (trajectory.mouse_y - trajectory.eye_y_correct_resp_area_center)**2 < self.resp_area_radius**2)
        
        is_in_cursor_not_response_area = is_in_cursor_area & \
                                        ~(is_in_incorrect_resp_area | is_in_correct_resp_area)
                                        
#        dwell_times = np.array([is_in_incorrect_resp_area.mean(), is_in_correct_resp_area.mean()])
        
        return pd.Series({'dwell_incorrect': is_in_incorrect_resp_area.mean(), 
                          'dwell_correct': is_in_correct_resp_area.mean(), 
                          'dwell_incorrect_not_cursor': is_in_incorrect_resp_area_not_cursor.mean(), 
                          'dwell_correct_not_cursor': is_in_correct_resp_area_not_cursor.mean(), 
                          'dwell_cursor': is_in_cursor_not_response_area.mean()})
    
    def append_is_saccade(self, norm_dynamics):
        # get indices of time points when the subjects 
        # saccade from incorrect to correct (or the other way around)        
        norm_dynamics = norm_dynamics. \
            groupby(level = ['subj_id', 'session_no', 'block_no', 'trial_no']).apply(self.get_is_saccade)
        return norm_dynamics

    def get_is_saccade(self, trajectory):
        # for each time step, check if eye_x changed sign 
        # possibly, later add another condition: eye_y is above center of the screen
        is_saccade_across_screen = np.append(False, abs(np.diff(np.sign(trajectory.eye_x))) > 1)
        is_eye_y_on_top = trajectory.eye_y > 1080*0.75
        trajectory['is_saccade'] = is_saccade_across_screen & is_eye_y_on_top
        return trajectory   
        
    def get_saccade_idx(self, choices, norm_dynamics, variables = []):
        norm_dynamics = self.append_is_saccade(norm_dynamics)
        cols = ['subj_id', 'block_no', 'trial_no'] + variables

        saccade_idx = norm_dynamics.groupby(by = cols). \
                apply(lambda x: x.loc[:, ['is_saccade']].reset_index(drop=True).T)
        choices['saccade_count'] = norm_dynamics.groupby(by=['subj_id', 'session_no', 'block_no', 'trial_no']).\
                apply(lambda traj: traj.is_saccade.sum())
        choices['saccade_idx'] = norm_dynamics.groupby(level = ['subj_id', 'session_no', 'block_no', 'trial_no']).\
                apply(lambda traj: np.nonzero(traj.is_saccade)[0])
                            
        return choices, norm_dynamics, saccade_idx
        
    def append_is_xflip(self, dynamics):
        dynamics = dynamics. \
            groupby(level = ['subj_id', 'session_no', 'block_no', 'trial_no']).apply(self.get_is_xflip)
        return dynamics
    
    def get_is_xflip(self, trajectory):
        trajectory['is_xflip'] = np.append(False, abs(np.diff(np.sign(trajectory.mouse_vx))) > 1)
        return trajectory
        
    def get_xflip_idx(self, choices, norm_dynamics, variables = []):
        norm_dynamics = self.append_is_xflip(norm_dynamics)
        cols = ['subj_id', 'session_no', 'block_no', 'trial_no'] + variables

        xflip_idx = norm_dynamics.groupby(by = cols). \
                apply(lambda x: x.loc[:, ['is_xflip']].reset_index(drop=True).T)

        choices['xflip_idx'] = norm_dynamics.groupby(level = ['subj_id', 'session_no', 'block_no', 'trial_no']).\
                apply(lambda traj: np.nonzero(abs(np.diff(np.sign(traj.mouse_vx))) > 1)[0])
                            
        return choices, norm_dynamics, xflip_idx
        
    def get_p_chosen_given_last_fixation(self, choices, norm_dynamics):
        choices['last_fixation'] = norm_dynamics.groupby(level = ['subj_id', 'session_no', 'block_no', 'trial_no']). \
                apply(lambda traj: 0 if traj.eye_x.values[-1]>0 else 180)
        return choices
    
    def append_mouse_eye_lag(self, choices, dynamics):
        choices['lag'] = dynamics.groupby(level = ['subj_id', 'session_no', 'block_no', 'trial_no']).\
            apply(lambda traj: np.correlate(traj.mouse_x, traj.eye_x, mode='full').argmax() \
            + 1 - len(traj))
        return choices
    
    def get_trajectory_fragment(self, trajectory, time_grid, mode='CoM', norm=False):
        # mode = 'CoM' or mode = 'initial'
        t = trajectory.timestamp
        
        if mode == 'CoM':
            center_idx = trajectory.idx_midline_d.ix[0]
        elif mode == 'initial':
            center_idx = int(np.where(trajectory.mouse_vx!=0)[0][0])
            
        center_timestamp = t[center_idx]
        
        centered_time_grid = center_timestamp + time_grid
            
        mouse_x_interp = np.interp(centered_time_grid, t.values, trajectory.mouse_x.values)
        mouse_y_interp = np.interp(centered_time_grid, t.values, trajectory.mouse_y.values)
        eye_x_interp = np.interp(centered_time_grid, t.values, trajectory.eye_x.values)
        eye_y_interp = np.interp(centered_time_grid, t.values, trajectory.eye_y.values)
        mouse_vx_interp = np.interp(centered_time_grid, t.values, trajectory.mouse_vx.values, 
                                    left=0, right=0)
        eye_vx_interp = np.interp(centered_time_grid, t.values, trajectory.eye_vx.values, 
                                  left=0, right=0)

#        mouse_ax_interp = np.interp(centered_time_grid, t.values, trajectory.mouse_ax.values, 
#                                    left=0, right=0)
#        eye_ax_interp = np.interp(centered_time_grid, t.values, trajectory.eye_ax.values, 
#                                  left=0, right=0)
        
        return pd.DataFrame(np.array([time_grid, mouse_x_interp, mouse_y_interp, 
                                      eye_x_interp, eye_y_interp,
                                      mouse_vx_interp, eye_vx_interp
#                                      ,mouse_ax_interp, eye_ax_interp
                                      ]).T)
                                      
    def get_dynamics_window(self, choices_com, dynamics_com, window=0.2, mode='CoM', norm=False):
        time_grid = np.linspace(-window/2.0, window/2.0, window/0.01 + 1)
        dynamics_com = dynamics_com.join(choices_com.idx_midline_d)
        dynamics_around_com = dynamics_com.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']). \
                        apply(lambda traj: self.get_trajectory_fragment(traj, time_grid, mode, norm))            
                
        dynamics_around_com.index = dynamics_around_com.index.droplevel(4)
        dynamics_around_com.columns = ['t', 'mouse_x', 'mouse_y', 'eye_x', 'eye_y',
                                               'mouse_vx', 'eye_vx'
#                                               , 'mouse_ax', 'eye_ax'
                                               ]
        dynamics_around_com = dynamics_around_com.join(choices_com.loc[:,['is_correct', 'is_com']])
        dynamics_around_com = dynamics_around_com.reset_index()
        dynamics_around_com['id'] = dynamics_around_com['subj_id'].map(str)  + '_' + \
                                    dynamics_around_com['session_no'].map(str) + '_' +\
                                    dynamics_around_com['block_no'].map(str) + '_' + \
                                    dynamics_around_com['trial_no'].map(str)
        return dynamics_around_com
    
    def get_delta_around_com(self, trajectory, variable, window=0.2, offset=None, mode='central'):   
        com_idx = trajectory.idx_midline_d.ix[0]
        com_timestamp = trajectory.timestamp[com_idx]
    
        if offset is None:
            if mode == 'central':
                offset = 0
            elif mode=='left':
                offset = -window/2.0
            elif mode=='right':
                offset = window/2.0

        left = max(com_timestamp + offset - window/2.0, trajectory.timestamp.min())
        right = min(com_timestamp + offset + window/2.0, trajectory.timestamp.max())
    
        var_interp = np.interp([left, right], trajectory.timestamp.values, 
                               trajectory[variable].values)
        return np.diff(var_interp)[0]

    def append_com_deltas(self, choices_com, dynamics_com, window=0.2, offset=None, mode='central'):    
        dynamics_com = dynamics_com.join(choices_com.idx_midline_d)
        choices_com['delta_eye_x_com_'+mode] = dynamics_com. \
                groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']). \
                apply(lambda traj: self.get_delta_around_com(traj, 'eye_x', window=window, mode=mode))
        choices_com['delta_ps_com_'+mode] = dynamics_com. \
                groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']). \
                apply(lambda traj: self.get_delta_around_com(traj, 'pupil_size', window=window, mode=mode))
        return choices_com
    
    def normalize_IT(self, choices):
        choices['initiation_time_norm'] = choices.initiation_time.groupby(level='subj_id'). \
                apply(lambda c: c/c.mean())
        return choices
    
    def get_mouse_eye_desync_rate(self, trajectory):      
#       is_in_sync is equal to 1 when mouse moves in the direction of eye_AOI
#       is_in_sync is equal to -1 when mouse moves away from eye_AOI        
#       is_in_sync is equal to 0 when eye isn't in response locations
        is_in_sync = np.sign(trajectory.mouse_vx) * trajectory.eye_AOI
#        is_in_sync = np.sign(trajectory.mouse_vx * trajectory.eye_x)
        
        # TODO: discard desync when mouse is within eye's AOI
        return len(is_in_sync[(is_in_sync==-1) & (trajectory.mouse_AOI != trajectory.eye_AOI)])/len(is_in_sync)

    def append_eye_AOI(self, dynamics):
        dynamics.loc[:, 'eye_AOI'] = 0

#        dynamics.loc[(dynamics.eye_x - self.x_lim/2.0)**2 + (dynamics.eye_y - self.y_lim/2)**2 < \
#                    self.center_AOI_radius**2, 'eye_AOI'] = 0
                    
        dynamics.loc[(dynamics.eye_x < -self.x_lim/2.0 + \
                    2*(self.resp_area_radius + self.resp_area_offset)) & \
                    (dynamics.eye_y > self.y_lim - \
                    2*(self.resp_area_radius + self.resp_area_offset)), 
                    'eye_AOI'] = -1

        dynamics.loc[(dynamics.eye_x > self.x_lim/2.0 - \
                    2*(self.resp_area_radius + self.resp_area_offset)) & \
                    (dynamics.eye_y > self.y_lim - \
                    2*(self.resp_area_radius + self.resp_area_offset)), 
                    'eye_AOI'] = 1
        return dynamics

    def append_mouse_AOI(self, dynamics):
        dynamics.loc[:, 'mouse_AOI'] = 0

#        dynamics.loc[(dynamics.eye_x - self.x_lim/2.0)**2 + (dynamics.eye_y - self.y_lim/2)**2 < \
#                    self.center_AOI_radius**2, 'eye_AOI'] = 0
                    
        dynamics.loc[(dynamics.mouse_x < -self.x_lim/2.0 + \
                    2*(self.resp_area_radius + self.resp_area_offset)) & \
                    (dynamics.mouse_y > self.y_lim - \
                    2*(self.resp_area_radius + self.resp_area_offset)), 
                    'mouse_AOI'] = -1

        dynamics.loc[(dynamics.mouse_x > self.x_lim/2.0 - \
                    2*(self.resp_area_radius + self.resp_area_offset)) & \
                    (dynamics.mouse_y > self.y_lim - \
                    2*(self.resp_area_radius + self.resp_area_offset)), 
                    'mouse_AOI'] = 1
        return dynamics
                    
    # calculate probability of xflip as a function of time t given that at t-k there was a saccade
#    def get_p_xflip(k = 0):
#        for t in range(1,51):
#        # first, get all the choices where there was a saccade at t-k
#        # for this, manually go over all the entries in choices, 
#        for choice in choices.saccade_idx:
#            print(choice)
#        choices[choices.saccade_.contains()]
        
        
        