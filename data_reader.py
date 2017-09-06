import pandas as pd

class DataReader:    
    def get_data(self, path, stim_viewing=False):
        filePath = path + '%s.txt'
        choicesFilePath = filePath % ('choices')
        choices = pd.read_csv(choicesFilePath, sep='\t')
        choices.set_index(['subj_id', 'session_no', 'block_no', 'trial_no'], inplace=True, drop=False)
        choices['is_correct'] = choices['direction'] == choices['response']
        choices.rename(columns={'reaction_time': 'initiation_time'}, inplace=True)
        
        dynamicsFilePath = filePath % ('dynamics')       
        dynamics = pd.read_csv(dynamicsFilePath, sep='\t')        
        dynamics.set_index(['subj_id', 'session_no', 'block_no', 'trial_no'], inplace=True, drop=False)
        
        choices = choices[~(choices.subj_id==702)]
        dynamics = dynamics[~(dynamics.subj_id==702)]
        
        if stim_viewing:
            stimFilePath = filePath % ('stim')       
            stim_viewing = pd.read_csv(stimFilePath, sep='\t')        
            stim_viewing.set_index(['subj_id', 'session_no', 'block_no', 'trial_no'], 
                                   inplace=True, drop=False)
            return choices, dynamics, stim_viewing        
        else:
            return choices, dynamics 