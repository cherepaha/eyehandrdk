import os

data_types = ['choices', 'dynamics', 'stim_viewing']

exp = 'HEM_exp_2'

out_folder = '../../data/%s/merged_raw/' % (exp)
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

for data_type in data_types:
    in_path = '../../data/%s/raw/' % (exp) + data_type 

    out_path = out_folder + data_type
    fout=open(out_path + '.txt','w+')
    for i, f in enumerate(os.listdir(in_path)):
        file_path=os.path.join(in_path,f)            
        if file_path.endswith('.txt'):
            f = open(file_path)
            if i!=0:
                # skip the header for the first row
                next(f) 
            for line in f:
                fout.write(line)
            f.close()
            print(file_path)
    fout.close()