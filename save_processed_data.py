import data_reader, data_preprocessor, os

def save_processed_data():
    reader = data_reader.DataReader()
    preprocessor = data_preprocessor.DataPreprocessor()

    choices, dynamics = reader.get_data()
    choices, dynamics = preprocessor.preprocess_data(choices, dynamics)
    
    path = '../data/processed/'
    if not os.path.exists(path):
        os.makedirs(path)
    choices.to_csv(path + 'choices.txt', sep='\t')
    dynamics.to_csv(path + 'dynamics.txt', sep='\t')

save_processed_data()