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
best_param_folder = '/home/erussek/projects/RNNs/best_hyper_params_sequential'

# Read in best hyper params...
best_param_dict = pickle.load(open(os.path.join(best_param_folder, 'best_hyper_params.pkl'), 'rb'))

# types of models...
model_names = ['LSTM', 'GRU', 'Transformer']
# model_names = ['LSTM']

part_names = ["Simulated_Only","Simulated_Only_Pred_Human", "Human_Only", "Simulated_and_Human"]
part_names = [ "Human_Only", "Simulated_and_Human"]
fix_unit_types = ['ID', 'all']

# train_seq_parts = ['fix_only', 'fix_and_choice']
train_seq_parts = ['fix_and_choice']


job_model_names = []
job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_attention_heads = []
job_layers = []
job_runs = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_train_names = []
job_full_names = []
job_tsps = []

# for simulated only -- find both the params that maximize simulated data performance 
# and also the params that maximize human performance

n_runs = 200 # go for 200 runs...
for run in range(20,n_runs):
    for fu in fix_unit_types:
        for model_name in model_names:
            for tsp in train_seq_parts:
                for part_name in part_names:
                    full_name = "{}_{}_{}_{}".format(tsp, part_name, model_name, fu)
                    job_full_names.append(full_name + '_run_{}'.format(run))
                    mbp = best_param_dict[full_name]
                    
                    job_runs.append(run)
                    job_tsps.append(tsp)
                    job_model_names.append(mbp['model_name'])
                    job_hidden_sizes.append(mbp['d_model'])
                    job_human_lrs.append(mbp['human_lr'])
                    job_attention_heads.append(mbp['n_head'])
                    job_layers.append(mbp['n_layers'])
                    
                    if part_name == 'Simulated_Only':
                        job_n_sim_seqs.append(1.5e6)
                    else:
                        job_n_sim_seqs.append(mbp['best_sim_num'])
                        
                    job_sim_lrs.append(mbp['sim_lr'])

                        
                    #elif part_name == "Simulated_and_Human":
                    #    this_full_name = "{}_{}_{}_{}".format(tsp, "Simulated_Only_Pred_Human", model_name, fu)
                    #    this_mbp = best_param_dict[this_full_name]
                    #    job_n_sim_seqs.append(this_mbp['best_sim_num'])
                    #    job_sim_lrs.append(this_mbp['sim_lr'])
                        
                    #else:
                    #    job_n_sim_seqs.append(this_mbp['best_sim_num'])
                    #    job_sim_lrs.append(mbp['sim_lr'])

                        
                    job_n_human_seqs.append(mbp['best_hum_num'])
                    job_train_names.append(part_name)             
                    

                    
#which_jobs = np.reshape(np.arange(2400), (int(2400/3),3))
which_jobs = np.reshape(np.arange(len(job_runs)), (int(len(job_runs)/3),3))

these_jobs = which_jobs[this_job_idx,:]

for job_idx in these_jobs:
    
    print(job_idx)
    
    run_idx = job_runs[job_idx]
    model_name = job_model_names[job_idx]
    d_model = job_hidden_sizes[job_idx]
    sim_lr = job_sim_lrs[job_idx]
    human_lr = job_human_lrs[job_idx]
    n_head = job_attention_heads[job_idx]
    n_layers = job_layers[job_idx]
    n_simulation_sequences_train = job_n_sim_seqs[job_idx]
    n_human_sequences_train = job_n_human_seqs[job_idx]
    train_name = job_train_names[job_idx]

    save_file_name = job_full_names[job_idx]

    main_as_fun(
        run_idx = job_runs[job_idx],
        model_name = job_model_names[job_idx],
        d_model = job_hidden_sizes[job_idx],
        sim_lr = job_sim_lrs[job_idx],
        human_lr = job_human_lrs[job_idx],
        n_head = job_attention_heads[job_idx],
        n_layers = job_layers[job_idx],
        train_seq_part = job_tsps[job_idx],
        n_simulation_sequences_train = job_n_sim_seqs[job_idx],
        n_human_sequences_train = job_n_human_seqs[job_idx],
        dropout = .1,
        on_cluster = True,
        save_folder_name = 'Sequential_Best_Param_Results_F',
        save_file_name = save_file_name)