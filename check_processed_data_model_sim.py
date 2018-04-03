import data_reader, data_preprocessor, data_analyser, trajectory_plotter
import pandas as pd
import numpy as np

path='../../data/HEM_sim/processed/'

dr = data_reader.DataReader()
dp = data_preprocessor.DataPreprocessor()
da = data_analyser.DataAnalyser()
tp = trajectory_plotter.TrajectoryPlotter()

choices, dynamics = dr.get_data(path=path, stim_viewing=False, sep='\t')
#choices = choices[choices.index.get_level_values('trial_no')<501]

traj = da.get_random_trajectory(choices[choices.is_com], dynamics)

tp.plot_trajectory_x(traj)

#choicesFilePath = path+'choices.txt'
#choices = pd.read_csv(choicesFilePath)

#choices, dynamics = dr.get_data(path=path, stim_viewing=False, sep=',', nrows=2000000)
