import data_reader, data_preprocessor, os

path='../../data/HEM_sim/'

dr = data_reader.DataReader()
dp = data_preprocessor.DataPreprocessor()

n_trials = 1000
trial_len = 4000

choices, dynamics = dr.get_data(path=path, stim_viewing=False, sep=',', nrows=n_trials*trial_len)
choices = choices[choices.index.get_level_values('trial_no')<n_trials+1]

dynamics = dp.preprocess_data(choices, dynamics, model_data=True)
choices = dp.get_mouse_and_gaze_measures(choices, dynamics, model_data=True)

processed_path = path + 'processed/'
if not os.path.exists(processed_path):
    os.makedirs(processed_path)
choices.to_csv(processed_path + 'choices.txt', sep='\t', na_rep='nan', float_format='%.4f')
dynamics.to_csv(processed_path + 'dynamics.txt', sep='\t', na_rep='nan', float_format='%.4f')
