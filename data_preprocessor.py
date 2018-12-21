from __future__ import division
import pandas as pd
import numpy as np
import derivative_calculator

class DataPreprocessor:
    x_lim = 1920
    y_lim = 1080
    
    com_threshold_x = 50
    com_threshold_y = 100

    late_com_y_threshold = 700
    
    # to determine exact response intiation,
    # threshold for distance travelled by mouse cursor (in pixels) during single movement
    # it is needed when calculating initation time
    it_distance_threshold = 100
    eye_v_threshold = 1000
    
    resp_AOI_radius = 200
    mouse_AOI_radius = 100
    incorrect_AOI_centre = [-(x_lim/2-resp_AOI_radius), (y_lim-resp_AOI_radius)]
    correct_AOI_centre = [(x_lim/2-resp_AOI_radius), (y_lim-resp_AOI_radius)] 
        
    index = ['subj_id', 'session_no', 'block_no', 'trial_no']
    
    def preprocess_data(self, choices, dynamics, resample=0, model_data=False):       
        if not model_data:
            # originally, EyeLink data has -32768.0 values in place when data loss occurred
            # we replace it with np.nan to be able to use numpy functions properly
            dynamics = dynamics.replace(dynamics.eye_x.min(), np.nan)
            
            dynamics = self.set_origin_to_start(dynamics)                   
        
        dynamics = self.shift_timeframe(dynamics)
        
        # flip trajectories for trials where direction == left
        # so that all correct trajectories go to the right
        dynamics.loc[choices.direction==180, ['mouse_x', 'eye_x']] *= -1
        
        if resample:
            dynamics = self.resample_trajectories(dynamics, n=resample)
        
        dc = derivative_calculator.DerivativeCalculator()
        dynamics = dc.append_diff(dynamics)
        dynamics = dc.append_derivatives(dynamics)
        
        dynamics['mouse_v'] = np.sqrt(dynamics.mouse_vx**2 + dynamics.mouse_vy**2 )
        dynamics['eye_v'] = np.sqrt(dynamics.eye_vx**2 + dynamics.eye_vy**2 )        
                          
        return dynamics  
    
    def get_hand_and_gaze_measures(self, choices, dynamics, stim_viewing=None, model_data=False):
        # TODO: get rid of extra index added as extra columns somewhere along the way (subj_id.1, ...)
        choices['is_correct'] = choices['direction'] == choices['response']        
        choices.response_time /= 1000.0        
        choices['xflips'] = dynamics.groupby(level=self.index).\
                                    apply(lambda traj: self.zero_cross_count(traj.mouse_vx.values))    
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_maxd))
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_midline_d))

        if model_data:
            choices['is_com'] = ((choices.midline_d > self.com_threshold_x))
        else:                
            choices['is_com'] = ((choices.midline_d > self.com_threshold_x) & \
                                    (choices.midline_d_y > self.com_threshold_y))
        choices['com_type'] = pd.cut(choices.midline_d_y, 
               bins=[0, self.late_com_y_threshold, self.y_lim], labels=['early', 'late'])

        choices['is_correct_init'] = choices['is_correct']
        choices.loc[choices.is_com, 'is_correct_init'] = (dynamics[choices.is_com]. \
                   groupby(level=self.index).apply(self.get_initial_decision))
        choices['is_double_com'] = choices.is_com & (choices.is_correct_init == choices.is_correct)

        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_hand_IT))
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_eye_IT))
        
        # initiation time during stimulus presentation is aligned at stimulus offset, so it is non-positive
        if not stim_viewing is None:
            choices['stim_hand_IT'] = stim_viewing.groupby(level=self.index).apply(self.get_stim_hand_IT)
            choices['stim_eye_IT'] = stim_viewing.groupby(level=self.index).apply(self.get_stim_eye_IT)
            # Comment next line for premature responses to have hand_IT = 0 regardless hand movements during stimulus viewing   
            # TODO: even after this, there are too many zero hand IT's
            # investigate the trajectories in Exp 2 where both hand_IT = 0 and stim_hand_IT = 0
            # Most are in subject 624
            choices.loc[choices.hand_IT==0, 'hand_IT'] = choices.loc[choices.hand_IT==0, 'stim_hand_IT']
            # this correction is also recommended for eye_IT
            choices.loc[choices.eye_IT==0, 'eye_IT'] = choices.loc[choices.eye_IT==0, 'stim_eye_IT']
                
        choices['ID_lag'] = choices.hand_IT - choices.eye_IT
        
        # TODO: figure out how to pass idx_midline_d to get_com_lag without adding a column to dynamics dataframe
        dynamics = dynamics.join(choices.idx_midline_d)
        choices = choices.join(dynamics[choices.is_com].groupby(level=self.index).apply(self.get_com_lag))        
        
        # We can also z-score within participant AND coherence level, the results remain the same
        # ['subj_id', 'coherence']
        z = lambda c: (c-np.nanmean(c))/np.nanstd(c)
        choices['hand_IT_z'] = choices.hand_IT.groupby(level='subj_id').apply(z)
        choices['eye_IT_z'] = choices.eye_IT.groupby(level='subj_id').apply(z)
        choices['ID_lag_z'] = choices.ID_lag.groupby(level='subj_id').apply(z)
        
        if not model_data:
            choices = self.append_dwell_times(choices, dynamics)
            choices['hand_IT_tertile'] = pd.qcut(choices['hand_IT'], 3, labels=[1, 2, 3])
            choices['eye_IT_tertile'] = pd.qcut(choices['eye_IT'], 3, labels=[1, 2, 3])
        
        return choices
    
    def exclude_trials(self, choices, dynamics, stim_viewing):        
        dynamics = dynamics[~choices.is_double_com]
        stim_viewing = stim_viewing[~choices.is_double_com]
        choices = choices[~choices.is_double_com]
        
        return choices, dynamics, stim_viewing
    
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

    def resample_trajectories(self, dynamics, n_steps=100):
        resampled_dynamics = dynamics.groupby(level=self.index).\
                                    apply(lambda traj: self.resample_trajectory(traj, n_steps=n_steps))
        resampled_dynamics.index = resampled_dynamics.index.droplevel(4)
        return resampled_dynamics
        
    def get_maxd(self, traj):
        alpha = np.arctan((traj.mouse_y.iloc[-1]-traj.mouse_y.iloc[0])/ \
                            (traj.mouse_x.iloc[-1]-traj.mouse_x.iloc[0]))
        d = (traj.mouse_x.values-traj.mouse_x.values[0])*np.sin(-alpha) + \
            (traj.mouse_y.values-traj.mouse_y.values[0])*np.cos(-alpha)
        if abs(d.min())>abs(d.max()):
            return pd.Series({'max_d': d.min(), 'idx_max_d': d.argmin()})
        else:
            return pd.Series({'max_d': d.max(), 'idx_max_d': d.argmax()})
        
    def get_midline_d(self, traj):
        mouse_x = traj.mouse_x.values
        is_final_point_positive = (mouse_x[-1]>0)
        
        midline_d = mouse_x.min() if is_final_point_positive else mouse_x.max()

        idx_midline_d = (mouse_x == midline_d).nonzero()[0][-1]
        midline_d_y = traj.mouse_y.values[idx_midline_d]
        return pd.Series({'midline_d': abs(midline_d), 
                          'idx_midline_d': idx_midline_d,
                          'midline_d_y': midline_d_y})

    def get_initial_decision(self, traj):    
        # after flipping, x>0 is correct
        return traj.mouse_x.values[np.argmax(abs(traj.mouse_x.values)>self.com_threshold_x)] > 0
    
    def zero_cross_count(self, x):
        return (abs(np.diff(np.sign(x)[np.nonzero(np.sign(x))]))>1).sum()

    def get_hand_IT(self, traj):
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
        if len(submovements):
            it = submovements.loc[submovements.distance.ge(self.it_distance_threshold ).idxmax()].on_t
        else:
            it = np.inf
        return pd.Series({'hand_IT': it, 'motion_time': traj.timestamp.max()-it})
    
    def get_eye_IT(self, traj):        
        v = traj.eye_v
        if ((v < self.eye_v_threshold) | (v.isnull())).all():
            # if eye stays at one location throughout the whole trial, 
            # it is supposedly fixated either at the center of the screen, or at the response location
            # in the former case, initation time is inf, in the latter case, initation time is 0
            # we detect which case is tru  by looking at first value of eye_x
            if abs(traj.eye_x.iloc[0]) < 100: 
                eye_IT = np.nan
                eye_initial_decision = 0
            else:
                eye_IT = 0
                eye_initial_decision = np.sign(traj.eye_x.iloc[0])
        else:
            eye_IT_idx = (v > self.eye_v_threshold).nonzero()[0][0]
            eye_IT = traj.timestamp.iloc[eye_IT_idx]
            eye_initial_decision = np.sign(traj.eye_x.iloc[eye_IT_idx+1])
        
        return pd.Series({'eye_IT': eye_IT, 'eye_initial_decision': eye_initial_decision})

    def get_stim_hand_IT(self, stim_traj):
        t = stim_traj.timestamp.values
        v = stim_traj.mouse_v.values
        if v[-1]:
            idx = np.where(v==0)[0][-1]+1 if len(v[v==0]) else 0
            IT = (t[idx] - t.max())
        else:
            IT = 0
        return IT
    
    def get_stim_eye_IT(self, stim_traj):
        t = stim_traj.timestamp.values
        v = stim_traj.eye_v.values
    
        idx = np.where(v<self.eye_v_threshold)[0][-1] if len(v[v<self.eye_v_threshold]) else 0
        return (t[idx] - t.max())
    
    def resample_trajectory(self, traj, n_steps):
        # Make the sampling time intervals regular
        n = np.arange(0, n_steps+1)
        t_regular = np.linspace(traj.timestamp.min(), traj.timestamp.max(), n_steps+1)
        mouse_x_interp = np.interp(t_regular, traj.timestamp.values, traj.mouse_x.values)
        mouse_y_interp = np.interp(t_regular, traj.timestamp.values, traj.mouse_y.values)
        eye_x_interp = np.interp(t_regular, traj.timestamp.values, traj.eye_x.values)
        eye_y_interp = np.interp(t_regular, traj.timestamp.values, traj.eye_y.values)
        pupil_size_interp = np.interp(t_regular, traj.timestamp.values, 
                                      traj.pupil_size.values)
        traj_interp = pd.DataFrame([n, t_regular, mouse_x_interp, mouse_y_interp, \
                                    eye_x_interp, eye_y_interp, pupil_size_interp]).transpose()
        traj_interp.columns = ['n', 'timestamp', 'mouse_x', 'mouse_y', 'eye_x', 'eye_y', 'pupil_size']
#        traj_interp.index = range(1,n_steps+1)
        return traj_interp
    
    def remove_blinks(self, x, fillna='extrapolate'):
        # number of time steps around blink to erase
        # 5 steps is 50 ms
        window = 5
        onsets = []
        offsets = []
        is_previous_x_nan = False
        
        # first, get onsets and offsets for each blink
        for i in range(0,len(x)):
            if np.isnan(x[i]):
                if not is_previous_x_nan:
                    is_previous_x_nan = True
                    onsets += [i]
                elif (i==len(x)-1):
                    offsets += [i]            
            elif is_previous_x_nan:
                offsets += [i]
                is_previous_x_nan = False
                
        if len(onsets) != len(offsets):
            raise ValueError('NaN onsets and offsets do not match!')
            
        # second, for every blink, drop 'window' data points immediately before and after the blink,
        # and fill in the blanks by extrapolating from the available data points (or zeros, depending on fillna parameter)
        for i in range(0,len(onsets)):
            onset = onsets[i]
            offset = offsets[i]        
            if fillna=='extrapolate':
                x[onset-window:int(np.floor((onset+offset)/2))] = x[onset-window-1]
                x[int(np.floor((onset+offset)/2)):offset+window] = x[offset+window+1]
            elif fillna=='zeros':
                x[onset-window:offset+window] = 0
            else:
                raise ValueError('Incorrect value of fillna parameter')
                    
        return x

    def get_com_lag(self, trajectory):
        t = trajectory.timestamp.values
        
        com_idx = int(trajectory.idx_midline_d.values[0])
        com_t = t[com_idx]
        com_direction = np.sign(trajectory.mouse_vx.iloc[com_idx + 1])

        v = self.remove_blinks(trajectory.eye_vx.values, fillna='zeros')
    
        # to simplify saccade identification, treat all sub-threshold velocity values as zeros
        # to simplify this further, we only care about the saccades that match the direction of CoM
        # so we can simply drop all velocities in the wrong direction to zero
        v[(abs(v)<self.eye_v_threshold) | (~(np.sign(v)==com_direction))] = 0

        onsets = []
        offsets = []
        is_previous_v_zero = True
        
        # next, we go through every value of v and extract saccade onsets and offsets    
        for i in range(0,len(v)):
            if v[i]!=0:
                if is_previous_v_zero:
                    is_previous_v_zero = False
                    onsets += [i]
                elif (i==len(v)-1):
                    offsets += [i]            
            elif (not is_previous_v_zero):
                offsets += [i]
                is_previous_v_zero = True
        
        if len(onsets) == 0:
            (com_saccade_idx, com_saccade_t, com_lag) = (np.nan, np.nan, np.nan)
        elif len(onsets) != len(offsets):
            raise ValueError('Saccade onsets and offsets do not match! Take a closer look at trial %s' 
                             % (str(trajectory.index.unique()[0])))
        else:
            # finally, check which saccade is closest in time to CoM
            closest_onset_idx = abs(np.array(onsets)-com_idx).argmin()
            closest_offset_idx = abs(np.array(offsets)-com_idx).argmin()
            
            closest_onset = onsets[closest_onset_idx]    
            closest_offset = offsets[closest_offset_idx]
            
            # we have found the saccade onset and saccade offset which are closest in time to CoM
            # let's find out which one of the two is closer
            # if saccade onset is closer, it's marked as onset of CoM saccade
            # if saccade offset is closer, then the onset of that saccade is marked as onset of CoM saccade
            com_saccade_idx = (closest_onset 
                               if abs(closest_onset - com_idx) < abs (closest_offset - com_idx) 
                               else onsets[closest_offset_idx])
            
            com_saccade_t = t[com_saccade_idx]            
            com_lag = (com_t - com_saccade_t)
        
        return pd.Series({'com_saccade_idx': com_saccade_idx,
                          'com_t': com_t,
                          'com_saccade_t': com_saccade_t, 
                          'com_lag':com_lag})
    
    def append_dwell_times(self, choices, dynamics):
        dwell_times = dynamics.groupby(by=self.index).apply(self.get_dwell_times)
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
    
        return choices

    def get_dwell_times(self, trajectory):
        """
        This is all assuming that correct option is always on the right-hand side of the screen (flipped trajectories)
        """
        is_in_cursor_area = (trajectory.eye_x - trajectory.mouse_x)**2 + \
                    (trajectory.eye_y - trajectory.mouse_y)**2 < self.mouse_AOI_radius**2
        is_in_incorrect_resp_area = (trajectory.eye_x - self.incorrect_AOI_centre[0])**2 + \
                    (trajectory.eye_y - self.incorrect_AOI_centre[1])**2 < self.resp_AOI_radius**2
        is_in_correct_resp_area = (trajectory.eye_x - self.correct_AOI_centre[0])**2 + \
                    (trajectory.eye_y - self.correct_AOI_centre[1])**2 < self.resp_AOI_radius**2
    
        is_in_incorrect_resp_area_not_cursor = (is_in_incorrect_resp_area) & \
                    ((trajectory.mouse_x - self.incorrect_AOI_centre[0])**2 + \
                    (trajectory.mouse_y - self.incorrect_AOI_centre[1])**2 < self.resp_AOI_radius**2)
    
        is_in_correct_resp_area_not_cursor = (is_in_correct_resp_area) & \
                    ((trajectory.mouse_x - self.correct_AOI_centre[0])**2 + \
                    (trajectory.mouse_y - self.correct_AOI_centre[1])**2 < self.resp_AOI_radius**2)
    
        is_in_cursor_not_response_area = is_in_cursor_area & ~(is_in_incorrect_resp_area | is_in_correct_resp_area)
    
        return pd.Series({'dwell_incorrect': is_in_incorrect_resp_area.mean(), 
                          'dwell_correct': is_in_correct_resp_area.mean(), 
                          'dwell_incorrect_not_cursor': is_in_incorrect_resp_area_not_cursor.mean(), 
                          'dwell_correct_not_cursor': is_in_correct_resp_area_not_cursor.mean(), 
                          'dwell_cursor': is_in_cursor_not_response_area.mean()})