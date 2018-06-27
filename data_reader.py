import pandas as pd

class DataReader:
    index = ['subj_id', 'session_no', 'block_no', 'trial_no']
    
    def get_data(self, path, stim_viewing=False, sep='\t', nrows=None, rename_vars=False):
        # rename_vars triggers renaming the variables to more readable versions
        filePath = path + '%s.txt'
        choicesFilePath = filePath % ('choices')
        choices = pd.read_csv(choicesFilePath, sep=sep).set_index(self.index, drop=True)
        
        dynamicsFilePath = filePath % ('dynamics')       
        dynamics = pd.read_csv(dynamicsFilePath, sep=sep, nrows=nrows, low_memory=True).set_index(self.index, drop=True)        
        
        if rename_vars:
            choices = choices.rename(columns={'mouse_IT': 'mouse IT', 'eye_IT': 'eye IT'})
            choices.loc[choices['is_correct'], 'choice'] = 'Correct'
            choices.loc[~choices['is_correct'], 'choice'] = 'Error'
            choices.loc[choices['is_com'], 'type'] = 'CoM'
            choices.loc[~choices['is_com'], 'type'] = 'non-CoM'
        
        if stim_viewing:
            stimFilePath = filePath % ('stim_viewing')       
            stim_viewing = pd.read_csv(stimFilePath, sep=sep).set_index(self.index, drop=True)
            
            return choices, dynamics, stim_viewing        
        else:
            return choices, dynamics 