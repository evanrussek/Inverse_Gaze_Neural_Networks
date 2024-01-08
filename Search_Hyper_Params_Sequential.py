# run through diff hyper parameters for training on sim (and testing on human)

import os
import numpy as np

from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    this_job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    this_job_idx = 10


## These are the varieties of model types, training data types, and training input representation types that we want to find the best params for
model_names = ['LSTM','GRU','Transformer']
train_seq_parts = ['fix_and_choice'] # only do fix and choice for now...

fix_unit_types = ['ID'] 

# These are the hyper-parameters that we want to vary / find the best of
hidden_sizes = np.array([32, 64, 128, 256, 512]) # removed 32, add X?

hidden_sizes = np.array([8]) # run 16 - mostly for transformer

sim_lrs = np.array([1e-5, 1e-4, 1e-3])
human_lrs_train = np.array([1e-5, 1e-4, 1e-3])
human_lrs_finetune = np.array([1e-5, 1e-4, 1e-3])
# human_lrs_finetune = np.array([1e-3, 1e-4])

# For the transformer only -

transformer_attention_heads = [4]
transformer_layers = [2]

dropout = .1 # this is just the default value - and is only relevant for tranformer models

# Number of runs to average over... 
n_runs = 5 # try to average oer 6 runs...

# vary model name, hidden size, learning rate, (transformer stuff if transformer), run
job_model_names = []
job_train_seq_parts = []
job_fix_units = []

job_runs = []
job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_attention_heads = []
job_layers = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_dropout = []
job_part_names = []


# Train on simulated data only -- set names for each of these

n_sim_seqs = 5e5
n_human_seqs = 0
part_name = "Simulated_Only"


for run in range(n_runs):
    
    for fu in fix_unit_types:
        for train_seq_part in train_seq_parts:
        
            for model_name in model_names:
                for hidden_size in hidden_sizes: # just run this for 512
                    for sim_lr in sim_lrs:

                        if model_name == 'Transformer':
                            for layer in transformer_layers:
                                for attention_heads in transformer_attention_heads:

                                    job_runs.append(run)
                                    job_model_names.append(model_name)
                                    job_hidden_sizes.append(hidden_size)
                                    job_sim_lrs.append(sim_lr)
                                    job_human_lrs.append(0)

                                    job_attention_heads.append(attention_heads)
                                    job_layers.append(layer)

                                    job_n_sim_seqs.append(n_sim_seqs)
                                    job_n_human_seqs.append(n_human_seqs)
                                    job_dropout.append(dropout)
                                    job_fix_units.append(fu)
                                    job_train_seq_parts.append(train_seq_part)
                                    
                                    job_part_names.append(part_name)

                        else:
                            job_runs.append(run)
                            job_model_names.append(model_name)
                            job_hidden_sizes.append(hidden_size)
                            job_sim_lrs.append(sim_lr)
                            job_human_lrs.append(0)

                            job_attention_heads.append(0)
                            job_layers.append(0)

                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)
                            job_dropout.append(dropout)
                            job_fix_units.append(fu)
                            job_train_seq_parts.append(train_seq_part)
                            
                            job_part_names.append(part_name)



     

# Train on human data only

n_sim_seqs = 0
n_human_seqs = 2e6
part_name = "Human_Only"

for run in range(n_runs):
    for fu in fix_unit_types:
        for train_seq_part in train_seq_parts:
            for model_name in model_names:
                for hidden_size in hidden_sizes:
                     for human_lr in human_lrs_train:
                        if model_name == 'Transformer':
                            for layer in transformer_layers:
                                for attention_heads in transformer_attention_heads:

                                    job_runs.append(run)
                                    job_model_names.append(model_name)
                                    job_hidden_sizes.append(hidden_size)
                                    job_sim_lrs.append(0)
                                    job_human_lrs.append(human_lr)

                                    job_attention_heads.append(attention_heads)
                                    job_layers.append(layer)

                                    job_n_sim_seqs.append(n_sim_seqs)
                                    job_n_human_seqs.append(n_human_seqs)
                                    job_dropout.append(dropout)
                                    job_fix_units.append(fu)
                                    job_train_seq_parts.append(train_seq_part)
                                    job_part_names.append(part_name)

                        else:
                            job_runs.append(run)
                            job_model_names.append(model_name)
                            job_hidden_sizes.append(hidden_size)
                            job_sim_lrs.append(0)
                            job_human_lrs.append(human_lr)

                            job_attention_heads.append(0)
                            job_layers.append(0)

                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)
                            job_dropout.append(dropout)
                            job_fix_units.append(fu)
                            job_train_seq_parts.append(train_seq_part)
                            
                            job_part_names.append(part_name)


# Train on simulated and finetune on human data


n_sim_seqs = 1e6 
n_human_seqs = 2e6
part_name = "Simulated_and_Human"

for run in range(n_runs):
    for fu in fix_unit_types:
        for train_seq_part in train_seq_parts:
            for model_name in model_names:
                for hidden_size in hidden_sizes:
                     for human_lr in human_lrs_finetune:
                        if model_name == 'Transformer':
                            for layer in transformer_layers:
                                for attention_heads in transformer_attention_heads:

                                    job_runs.append(run)
                                    job_model_names.append(model_name)
                                    job_hidden_sizes.append(hidden_size)
                                    job_sim_lrs.append(.001)
                                    job_human_lrs.append(human_lr)

                                    job_attention_heads.append(attention_heads)
                                    job_layers.append(layer)

                                    job_n_sim_seqs.append(n_sim_seqs)
                                    job_n_human_seqs.append(n_human_seqs)

                                    job_dropout.append(dropout)
                                    job_fix_units.append(fu)
                                    job_train_seq_parts.append(train_seq_part)
                                    job_part_names.append(part_name)


                        else:
                            job_runs.append(run)
                            job_model_names.append(model_name)
                            job_hidden_sizes.append(hidden_size)
                            job_sim_lrs.append(.001) # just use a neutral sim learning rate for this...
                            job_human_lrs.append(human_lr)

                            job_attention_heads.append(0)
                            job_layers.append(0)

                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)

                            job_dropout.append(dropout)
                            job_fix_units.append(fu)
                            job_train_seq_parts.append(train_seq_part)

                            job_part_names.append(part_name)

n_jobs = len(job_runs)                            
which_jobs = np.reshape(np.arange(n_jobs), (int(np.ceil(n_jobs/10)),10))
these_jobs = which_jobs[this_job_idx,:]

for job_idx in these_jobs:                    

    # grab job info for purposes of saving file
    run_idx = job_runs[job_idx]
    model_name = job_model_names[job_idx]
    d_model = job_hidden_sizes[job_idx]
    sim_lr = job_sim_lrs[job_idx]
    human_lr = job_human_lrs[job_idx]
    n_head = job_attention_heads[job_idx]
    n_layers = job_layers[job_idx]
    n_simulation_sequences_train = job_n_sim_seqs[job_idx]
    n_human_sequences_train = job_n_human_seqs[job_idx]
    dropout = job_dropout[job_idx]
    fu = job_fix_units[job_idx]
    train_seq_part = job_train_seq_parts[job_idx]
    part_name = job_part_names[job_idx]

    save_file_name = '{}_{}_{}_run_{}_model_name_{}_d_model_{}_sim_lr_{}_human_lr_{}_n_head_{}_n_layers_{}'.format(part_name,train_seq_part, fu, run_idx, model_name, d_model, sim_lr, human_lr, n_head, n_layers)

    main_as_fun(
        run_idx = job_runs[job_idx],
        model_name = job_model_names[job_idx],
        d_model = job_hidden_sizes[job_idx],
        sim_lr = job_sim_lrs[job_idx],
        human_lr = job_human_lrs[job_idx],
        n_head = job_attention_heads[job_idx],
        n_layers = job_layers[job_idx],
        train_seq_part = job_train_seq_parts[job_idx],
        fix_unit = job_fix_units[job_idx],
        n_simulation_sequences_train = job_n_sim_seqs[job_idx],
        n_human_sequences_train = job_n_human_seqs[job_idx],
        dropout = job_dropout[job_idx],
        on_cluster = True,
        save_folder_name = 'Hyper_Param_Search_Sequential_F',
        save_file_name = save_file_name)