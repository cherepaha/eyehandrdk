import pandas as pd
import random

class DataReader:    
    def get_data(self, path, stim_viewing=False, test_mode=False):
        filePath = path + '%s.txt'
        choicesFilePath = filePath % ('choices')
        choices = pd.read_csv(choicesFilePath, sep='\t')
        choices.set_index(['subj_id', 'session_no', 'block_no', 'trial_no'], inplace=True, drop=False)
        choices['is_correct'] = choices['direction'] == choices['response']
        choices.rename(columns={'reaction_time': 'initiation_time'}, inplace=True)
        
        dynamicsFilePath = filePath % ('dynamics')       
        dynamics = pd.read_csv(dynamicsFilePath, sep='\t')        
        dynamics.set_index(['subj_id', 'session_no', 'block_no', 'trial_no'], inplace=True, drop=False)
        
        rows = sorted(random.sample(list(choices.index), 100))
        if test_mode:            
            choices = choices.loc[rows]
            dynamics = dynamics.loc[rows]
        
        if stim_viewing:
            stimFilePath = filePath % ('stim')       
            stim_viewing = pd.read_csv(stimFilePath, sep='\t')        
            stim_viewing.set_index(['subj_id', 'session_no', 'block_no', 'trial_no'], 
                                   inplace=True, drop=False)
            if test_mode:
                stim_viewing = stim_viewing.loc[rows]
            
            return choices, dynamics, stim_viewing        
        else:
            return choices, dynamics 