import data_reader, data_preprocessor, os

dr = data_reader.DataReader()
dp = data_preprocessor.DataPreprocessor()

choices, dynamics, stim_viewing = dr.get_data(path='../../data/HEM_exp_1/merged_raw/', 
                                                  stim_viewing=True, test_mode=False)
dynamics = dp.preprocess_data(choices, dynamics)
stim_viewing = dp.preprocess_data(choices, stim_viewing)

choices = dp.get_mouse_and_gaze_measures(choices, dynamics)

#
#path = '../../data/HEM_exp_1/processed_test/'
#if not os.path.exists(path):
#    os.makedirs(path)
#choices.to_csv(path + 'choices.txt', sep='\t')
#dynamics.to_csv(path + 'dynamics.txt', sep='\t')
#
