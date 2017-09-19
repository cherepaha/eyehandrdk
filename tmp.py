import data_reader, data_preprocessor, os


reader = data_reader.DataReader()
preprocessor = data_preprocessor.DataPreprocessor()

choices, dynamics, stim_viewing = reader.get_data(path='../../data/HEM_exp_1/merged_raw/', stim_viewing=True)
#delta_t = dynamics.groupby(level=['subj_id', 'block_no', 'trial_no']).apply(lambda d: d.timestamp[1:]-d.timestamp[:-1])
#delta_t = dynamics.iloc[1:, dynamics.columns.get_loc('timestamp')] - dynamics.iloc[:-1, dynamics.columns.get_loc('timestamp')]

choices, dynamics = preprocessor.preprocess_data(choices, dynamics)

#path = '../data/processed/'
#if not os.path.exists(path):
#    os.makedirs(path)
#choices.to_csv(path + 'choices.txt', sep='\t')
#dynamics.to_csv(path + 'dynamics.txt', sep='\t')

