# train at best params control

import os
import numpy as np
import pickle

from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    this_job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    this_job_idx = 3

    
## Make folder to save best params
best_param_folder = '/home/erussek/projects/RNNs/best_hyper_params_control'

# Read in best hyper params...
best_param_dict = pickle.load(open(os.path.join(best_param_folder, 'best_hyper_params.pkl'), 'rb'))

# types of models...
part_names = ["Simulated_Only","Simulated_Only_Pred_Human", "Human_Only", "Simulated_and_Human"]
fix_unit_types = ['ID', 'sum', 'prop', 'all']

job_runs = []
job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_fix_units = []
job_part_names = []
job_full_names = []
# job_train_names = []

# for simulated only -- find both the params that maximize simulated data performance 
# and also the params that maximize human performance

n_runs = 200
for run in range(n_runs):
# store all parameters and loss curves... 
    for fu in fix_unit_types:
        for part_name in part_names:

            full_name = "{}_{}".format(part_name, fu)
            job_full_names.append(full_name + '_run_{}'.format(run))
            mbp = best_param_dict[full_name]

            job_runs.append(run)
            job_hidden_sizes.append(mbp['d_model'])
            job_sim_lrs.append(mbp['sim_lr'])
            job_human_lrs.append(mbp['human_lr'])
            job_fix_units.append(fu)

            if part_name == 'Simulated_Only':
                job_n_sim_seqs.append(1e6)
            else:

                job_n_sim_seqs.append(mbp['best_sim_num'])

            job_n_human_seqs.append(mbp['best_hum_num'])
            # job_train_names.append(part_name)

                    
which_jobs = np.reshape(np.arange(3200), (int(np.ceil(3200/8)),8))
these_jobs = which_jobs[this_job_idx,:]

for job_idx in these_jobs:
    
    print(job_idx)
    

    save_file_name = job_full_names[job_idx]
    
    main_as_fun(
        run_idx = job_runs[job_idx],
        fix_unit = job_fix_units[job_idx],
        model_name = 'MLP',
        train_seq_part = 'fix_only',
        d_model = job_hidden_sizes[job_idx],
        sim_lr = job_sim_lrs[job_idx],
        human_lr = job_human_lrs[job_idx],
        n_simulation_sequences_train = job_n_sim_seqs[job_idx],
        n_human_sequences_train = job_n_human_seqs[job_idx],
        dropout = .1,
        on_cluster = True,
        save_folder_name = 'Control_Best_Param_Results',
        save_file_name = save_file_name)