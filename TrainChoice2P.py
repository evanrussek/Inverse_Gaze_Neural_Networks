import os
import numpy as np
import pickle

# %run main_as_fun.py
from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    this_job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    this_job_idx = 1
    
part_names = ["Simulated_Only", "Human_Only", "Simulated_and_Human"]
part_names = ["Human_Only"]

n_sim_seqs = [0]
n_hum_seqs = [3e5]



job_runs = []
job_full_names = []
job_n_sim_seqs = []
job_n_human_seqs = []

n_runs = 400
for run in range(n_runs):
    for part_idx, part_name in enumerate(part_names):
        full_name = "{}".format(part_name)
        job_full_names.append(full_name + '_run_{}'.format(run))
        job_runs.append(run)
        job_n_sim_seqs.append(n_sim_seqs[part_idx])
        job_n_human_seqs.append(n_hum_seqs[part_idx])

        


# run 400 jobs...
which_jobs = np.reshape(np.arange(400), (int(400/10),10))
these_jobs = which_jobs[this_job_idx,:]

for job_idx in these_jobs:

    print(job_idx)

    save_file_name = job_full_names[job_idx]

    main_as_fun(
        run_idx = job_runs[job_idx],
        model_name = 'Choice2P',
        sim_lr = .001,
        human_lr = .001,
        train_seq_part = 'choice_only',
        n_simulation_sequences_train = job_n_sim_seqs[job_idx],
        n_human_sequences_train = job_n_human_seqs[job_idx],
        on_cluster = True,
        save_folder_name = 'Choice_Only_2P_Results_HO2',
        save_file_name = save_file_name)