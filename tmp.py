import data_reader, data_preprocessor, trajectory_plotter, os
import random

dr = data_reader.DataReader()
dp = data_preprocessor.DataPreprocessor()
tp = trajectory_plotter.TrajectoryPlotter()

index = ['subj_id', 'session_no', 'block_no', 'trial_no']

choices, dynamics, stim_viewing = dr.get_data(path='../../data/HEM_exp_1/merged_raw/', 
                                                  stim_viewing=True, test_mode=False)
dynamics = dp.preprocess_data(choices, dynamics)
stim_viewing = dp.preprocess_data(choices, stim_viewing)

traj = dynamics.loc[random.sample(list(choices.index), 1)]
tp.plot_trajectory_x(traj, v=True)

early_rt = stim_viewing.groupby(level=index).apply(
        lambda traj: traj.timestamp.max()-traj.timestamp[traj.mouse_vx==0].iloc[-1])
    
choices[early_rt>0].groupby('coherence').count()

early_rt[early_rt>0].hist(bins=20)

#
#path = '../../data/HEM_exp_1/processed_test/'
#if not os.path.exists(path):
#    os.makedirs(path)
#choices.to_csv(path + 'choices.txt', sep='\t')
#dynamics.to_csv(path + 'dynamics.txt', sep='\t')
#


#import data_reader, data_preprocessor, os
#
#dr = data_reader.DataReader()
#dp = data_preprocessor.DataPreprocessor()
#
#choices, dynamics, stim_viewing = dr.get_data(path='../../data/HEM_exp_1/merged_raw/', 
#                                                  stim_viewing=True, test_mode=False)
#dynamics = dp.preprocess_data(choices, dynamics)
#stim_viewing = dp.preprocess_data(choices, stim_viewing)
#
#choices = dp.get_mouse_and_gaze_measures(choices, dynamics)
#
