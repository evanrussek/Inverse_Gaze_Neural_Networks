import os
import numpy as np

from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    job_idx = 10

# loop through:

# model type, hidden size, learning rate (sim data), transformer attention heads, trans. layers, runs

# types of models...

### Train on these vals... 
hidden_sizes = np.array([32, 64, 128, 256])
sim_lrs = np.array([1e-5, 1e-4, 1e-3])
human_lrs_train = np.array([1e-5, 1e-4, 1e-3])
human_lrs_finetune = np.array([1e-5, 1e-4, 1e-3])

# fixation types
fix_units = ['ID', 'sum', 'prop', 'all']

job_runs = []
job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_fix_units = []
job_part_names = []


n_sim_seqs = 5e5
n_human_seqs = 0
part_name = "Simulated_Only"

n_runs = 2

for run in range(n_runs):
    for fu in fix_units:        
        for hidden_size in hidden_sizes: # just run this for 512
            for sim_lr in sim_lrs:
                job_runs.append(run)
                job_fix_units.append(fu)
                job_hidden_sizes.append(hidden_size)
                job_sim_lrs.append(sim_lr)
                job_human_lrs.append(0)
                job_n_sim_seqs.append(n_sim_seqs)
                job_n_human_seqs.append(n_human_seqs)
                job_part_names.append(part_name)

                    

n_sim_seqs = 0
n_human_seqs = 1e6
part_name = "Human_Only"


for run in range(n_runs):
    for fu in fix_units:        
        for hidden_size in hidden_sizes: # just run this for 512
            for human_lr in human_lrs_train:
                job_runs.append(run)
                job_fix_units.append(fu)
                job_hidden_sizes.append(hidden_size)
                job_sim_lrs.append(0)
                job_human_lrs.append(human_lr)
                job_n_sim_seqs.append(n_sim_seqs)
                job_n_human_seqs.append(n_human_seqs)
                job_part_names.append(part_name)

                    
n_sim_seqs = 3e5 # what's the good pretrain val?
n_human_seqs = 1e6
part_name = "Simulated_and_Human"

for run in range(n_runs):
    for fu in fix_units:  
        for hidden_size in hidden_sizes:
            for human_lr in human_lrs_finetune:
                job_runs.append(run)
                job_fix_units.append(fu)
                job_hidden_sizes.append(hidden_size)
                job_sim_lrs.append(.001)
                job_human_lrs.append(human_lr)
                job_n_sim_seqs.append(n_sim_seqs)
                job_n_human_seqs.append(n_human_seqs)
                job_part_names.append(part_name)


run_idx = job_runs[job_idx]
d_model = job_hidden_sizes[job_idx]
sim_lr = job_sim_lrs[job_idx]
human_lr = job_human_lrs[job_idx]
n_simulation_sequences_train = job_n_sim_seqs[job_idx]
n_human_sequences_train = job_n_human_seqs[job_idx]
fix_unit = job_fix_units[job_idx]
part_name = job_part_names[job_idx]


save_file_name = '{}_run_{}_fixunit_{}_d_model_{}_sim_lr_{}_human_lr_{}'.format(part_name,run_idx, fix_unit, d_model, sim_lr, human_lr)

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
    save_folder_name = 'Hyper_Param_Search_Control',
    save_file_name = save_file_name)