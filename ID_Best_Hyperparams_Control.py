# import packages

import numpy as np
import os
import pickle

# create folder to store best parameters
best_param_folder = '/home/erussek/projects/RNNs/best_hyper_params_control'
if not os.path.exists(best_param_folder):
    os.mkdir(best_param_folder)
best_param_dict = {}

# functions to generate loss curves for setting and get best parameters
def load_results(run_idx, part_name, fix_unit, d_model, sim_lr, human_lr):
    
    to_save_folder = '/scratch/gpfs/erussek/RNN_project/Hyper_Param_Search_Control'
    res_name_full = '{}_run_{}_fixunit_{}_d_model_{}_sim_lr_{}_human_lr_{}'.format(part_name,run_idx, fix_unit, d_model, sim_lr, human_lr)
    param_dict = {'part_name': part_name, 'fu': fix_unit, 'd_model':d_model, 'sim_lr':sim_lr, 'human_lr':human_lr}
    res_file_name = res_name_full + '.pickle'
    res_full_file_name = os.path.join(to_save_folder, res_file_name)
    file = open(res_full_file_name, 'rb')
    res = pickle.load(file)
    return res, param_dict

def load_results_all_runs(part_name, fix_unit, d_model, sim_lr, human_lr, n_runs = 2):
    
    results_list = []
    
    for run_idx in range(n_runs):
        res, param_dict = load_results(run_idx, part_name, fix_unit, d_model, sim_lr, human_lr)
        results_list.append(res)
        
    return results_list, param_dict

def get_learning_curve(part_name, fix_unit, d_model, sim_lr, human_lr, n_runs = 2, which_loss = 'simulation_loss_results'): 
    
    results_list, param_dict = load_results_all_runs(part_name, fix_unit, d_model, sim_lr, human_lr, n_runs = n_runs)
    
    loss_results_by_run = np.array([res[which_loss] for res in results_list])
    
    return np.mean(loss_results_by_run, axis=0), np.std(loss_results_by_run, axis=0)/np.sqrt(n_runs), results_list[0]['train_sequence_number'], results_list[0]['simulation_sequence_number'], results_list[0]['human_sequence_number'], param_dict

def get_best_params(res_losses, res_params, simulation_sequence_number, human_sequence_number):
    loss_arr = np.array(res_losses)
    
    min_val = np.min(loss_arr)
    
    min_flat_idx = np.argmin(loss_arr)
    (min_train_setting_idx,min_train_num_idx) = divmod(min_flat_idx, loss_arr.shape[1])

    best_params = res_params[min_train_setting_idx]
    
    best_params['best_sim_num'] = simulation_sequence_number[min_train_setting_idx][min_train_num_idx]
    best_params['best_hum_num'] = human_sequence_number[min_train_setting_idx][min_train_num_idx]

    best_params['min_loss'] = min_val
    
    return best_params

### Train on these vals... 
hidden_sizes = np.array([32, 64, 128, 256])
sim_lrs = np.array([1e-5, 1e-4, 1e-3])
human_lrs_train = np.array([1e-5, 1e-4, 1e-3])
human_lrs_finetune = np.array([1e-5, 1e-4, 1e-3])

n_runs = 2

part_names = ["Simulated_Only", "Human_Only", "Simulated_and_Human"]
# fixation types
fix_unit_types = ['ID', 'sum', 'prop', 'all']

# get loss curve for each model / train_seq_part / model_name / part_name

res_params = {}
res_losses = {}
res_human_seq_nums = {}
res_sim_seq_nums = {}

# store all parameters and loss curves... 
for fu in fix_unit_types:
    for part_name in part_names:
            full_name = "{}_{}".format(part_name, fu)
            res_params[full_name] = []
            res_losses[full_name] = []
            res_human_seq_nums[full_name] = []
            res_sim_seq_nums[full_name] = []

            if part_name == 'Simulated_Only':
                full_name = "{}_{}".format('Simulated_Only_Pred_Human', fu)
                res_params[full_name] = []
                res_losses[full_name] = []
                res_human_seq_nums[full_name] = []
                res_sim_seq_nums[full_name] = []



# store all parameters and loss curves... 
for fix_unit in fix_unit_types:
    for part_name in part_names:
        for d_model in hidden_sizes:

            if part_name == 'Simulated_Only':

                for sim_lr in sim_lrs:

                    human_lr = 0
                    mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name, fix_unit, d_model, sim_lr, human_lr, n_runs = 2, which_loss = 'simulation_loss_results')
                    full_name = "{}_{}".format(part_name, fix_unit)

                    res_losses[full_name].append(mean_loss)
                    res_params[full_name].append(this_params)
                    res_human_seq_nums[full_name].append(human_sequence_number)
                    res_sim_seq_nums[full_name].append(simulation_sequence_number)
                    
                    
                    # get the human loss...
                    mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name, fix_unit, d_model, sim_lr, human_lr, n_runs = 2, which_loss = 'human_loss_results')
                    full_name = "{}_{}".format('Simulated_Only_Pred_Human', fix_unit)
                    res_losses[full_name].append(mean_loss)
                    res_params[full_name].append(this_params)
                    res_human_seq_nums[full_name].append(human_sequence_number)
                    res_sim_seq_nums[full_name].append(simulation_sequence_number)
                    
                    
            else: # not Sim Only
                        
                if part_name == 'Human_Only':
                    sim_lr = 0
                    these_human_lrs = human_lrs_train
                    n_runs = 2

                else:
                    sim_lr = .001
                    these_human_lrs = human_lrs_finetune
                    n_runs = 3
                    

                for human_lr in these_human_lrs:
                        mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name, fix_unit, d_model, sim_lr, human_lr, n_runs = n_runs, which_loss = 'human_loss_results')
                        full_name = "{}_{}".format(part_name, fix_unit)

                        res_losses[full_name].append(mean_loss)
                        res_params[full_name].append(this_params)
                        res_human_seq_nums[full_name].append(human_sequence_number)
                        res_sim_seq_nums[full_name].append(simulation_sequence_number)
                    

best_params_control = {}

for fix_unit in fix_unit_types:
    for part_name in part_names:
                
        full_name = "{}_{}".format(part_name, fix_unit)

        these_best_params = get_best_params(res_losses[full_name],res_params[full_name], res_sim_seq_nums[full_name], res_human_seq_nums[full_name])
        best_params_control[full_name] = these_best_params

        if part_name == 'Simulated_Only':
            full_name = "{}_{}".format('Simulated_Only_Pred_Human', fix_unit)
            these_best_params = get_best_params(res_losses[full_name],res_params[full_name], res_sim_seq_nums[full_name], res_human_seq_nums[full_name])
            best_params_control[full_name] = these_best_params

# save
f = open(os.path.join(best_param_folder, "best_hyper_params.pkl"),"wb")

# write the python object (dict) to pickle file
pickle.dump(best_params_control,f)

# close file
f.close()            
