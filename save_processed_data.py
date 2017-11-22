import data_reader, data_preprocessor, os

def save_processed_data(path):
    dr = data_reader.DataReader()
    dp = data_preprocessor.DataPreprocessor()
    
    choices, dynamics, stim_viewing = dr.get_data(path=path+'merged_raw/', stim_viewing=True)
    dynamics = dp.preprocess_data(choices, dynamics)
    stim_viewing = dp.preprocess_data(choices, stim_viewing)
    
    choices = dp.get_mouse_and_gaze_measures(choices, dynamics, stim_viewing)
    
    processed_path = path + 'processed/'
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    choices.to_csv(processed_path + 'choices.txt', sep='\t', na_rep='nan')
    dynamics.to_csv(processed_path + 'dynamics.txt', sep='\t', na_rep='nan')
    stim_viewing.to_csv(processed_path + 'stim_viewing.txt', sep='\t', na_rep='nan')

save_processed_data(path='../../data/HEM_exp_1/')
save_processed_data(path='../../data/HEM_exp_2/')