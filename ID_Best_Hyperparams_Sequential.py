# import packages


import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# create folder to store best parameters
best_param_folder = '/home/erussek/projects/RNNs/best_hyper_params_sequential'
if not os.path.exists(best_param_folder):
    os.mkdir(best_param_folder)

# functions to generate loss curves for setting and get best parameters
def load_results(run_idx, part_name,train_seq_part, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers):
    
    to_save_folder = '/scratch/gpfs/erussek/RNN_project/Hyper_Param_Search_Sequential_F'
    res_name_full = '{}_{}_{}_run_{}_model_name_{}_d_model_{}_sim_lr_{}_human_lr_{}_n_head_{}_n_layers_{}'.format(part_name,train_seq_part, fu, run_idx, model_name, d_model, sim_lr, human_lr, n_head, n_layers)
    param_dict = {'part_name': part_name, 'train_seq_part': train_seq_part, 'fu': fu, 'model_name':model_name, 'd_model':d_model, 'sim_lr':sim_lr, 'human_lr':human_lr, 'n_head':n_head, 'n_layers':n_layers}
    res_file_name = res_name_full + '.pickle'
    res_full_file_name = os.path.join(to_save_folder, res_file_name)
    file = open(res_full_file_name, 'rb')
    res = pickle.load(file)
    return res, param_dict

def load_results_all_runs(part_name,train_seq_part, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = 2):
    
    results_list = []
    
    for run_idx in range(n_runs):
        res, param_dict = load_results(run_idx, part_name,train_seq_part, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers)
        results_list.append(res)
        
    return results_list, param_dict

def get_learning_curve(part_name,train_seq_part, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = 1, smooth_loss = True, which_loss = 'simulation_loss_results'): 
    
    results_list, param_dict = load_results_all_runs(part_name,train_seq_part, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = n_runs)
    
    loss_results_by_run = np.array([res[which_loss] for res in results_list])
    
    mn_loss = np.mean(loss_results_by_run, axis=0)
    
    if smooth_loss:
        mn_loss = gaussian_filter(mn_loss, 3)
    
    return mn_loss, np.std(loss_results_by_run, axis=0)/np.sqrt(n_runs), results_list[0]['train_sequence_number'], results_list[0]['simulation_sequence_number'], results_list[0]['human_sequence_number'], param_dict

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


## These are the varieties of model types, training data types, and training input representation types that we want to find the best params for
model_names = ['LSTM','GRU','Transformer']
train_seq_parts = ['fix_only', 'fix_and_choice']
train_seq_parts = ['fix_and_choice']

# fix_unit_types = ['ID', 'all']
fix_unit_types = ['ID', 'all']

# These are the hyper-parameters that we want to vary / find the best of
hidden_sizes = np.array([8, 16, 32,64, 128, 256, 512])


sim_lrs = np.array([1e-3])
human_lrs_train = np.array([1e-5, 1e-4, 1e-3]) # just set this to 1e-3?
human_lrs_finetune = np.array([1e-5, 1e-4, 1e-3]) # just set this to 1e-3?


# For the transformer only
transformer_attention_heads = [4]
transformer_layers = [2]

part_names = ["Simulated_Only", "Human_Only", "Simulated_and_Human"]
part_names = ["Human_Only", "Simulated_and_Human"]

# get loss curve for each model / train_seq_part / model_name / part_name

res_params = {}
res_losses = {}
res_human_seq_nums = {}
res_sim_seq_nums = {}

for fu in fix_unit_types:
    for tsp in train_seq_parts:
        for part_name in part_names:
            for model_name in model_names:
                full_name = "{}_{}_{}_{}".format(tsp, part_name, model_name, fu)
                res_params[full_name] = []
                res_losses[full_name] = []
                res_human_seq_nums[full_name] = []
                res_sim_seq_nums[full_name] = []

                if part_name == 'Simulated_Only':
                    full_name = "{}_{}_{}_{}".format(tsp, 'Simulated_Only_Pred_Human', model_name, fu)
                    res_params[full_name] = []
                    res_losses[full_name] = []
                    res_human_seq_nums[full_name] = []
                    res_sim_seq_nums[full_name] = []
            

# loop through all models and find best parameters 
for fu in fix_unit_types:
    for model_name in model_names:
        for tsp in train_seq_parts:
            for part_name in part_names:
            
                if part_name == 'Simulated_Only':
                

                    # loop through params that we we want to max over...
                    for d_model in hidden_sizes:


                            n_runs = 2
                            for sim_lr in sim_lrs:

                                human_lr = 0

                                if model_name == 'Transformer':
                                    for n_layers in transformer_layers:
                                        for n_head in transformer_attention_heads:

                                            # get the simulation loss...
                                            mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name,tsp, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = 2, which_loss = 'simulation_loss_results')
                                            full_name = "{}_{}_{}_{}".format(tsp, part_name, model_name, fu)

                                            res_losses[full_name].append(mean_loss)
                                            res_params[full_name].append(this_params)

                                            res_human_seq_nums[full_name].append(human_sequence_number)
                                            res_sim_seq_nums[full_name].append(simulation_sequence_number)


                                            # get the human loss...
                                            mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name,tsp, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = 2, which_loss = 'human_loss_results')
                                            full_name = "{}_{}_{}_{}".format(tsp, 'Simulated_Only_Pred_Human', model_name, fu)
                                            res_losses[full_name].append(mean_loss)
                                            res_params[full_name].append(this_params)

                                            res_human_seq_nums[full_name].append(human_sequence_number)
                                            res_sim_seq_nums[full_name].append(simulation_sequence_number)

        
                                else: # not transformer
                                    n_head = 0
                                    n_layers = 0   

                                    # get the simulation loss...
                                    mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name,tsp, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = 2, which_loss = 'simulation_loss_results')
                                    full_name = "{}_{}_{}_{}".format(tsp, part_name, model_name, fu)

                                    res_losses[full_name].append(mean_loss)
                                    res_params[full_name].append(this_params)

                                    res_human_seq_nums[full_name].append(human_sequence_number)
                                    res_sim_seq_nums[full_name].append(simulation_sequence_number)

                                    # get the human loss...
                                    mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name,tsp, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = 2, which_loss = 'human_loss_results')
                                    full_name = "{}_{}_{}_{}".format(tsp, 'Simulated_Only_Pred_Human', model_name, fu)
                                    res_losses[full_name].append(mean_loss)
                                    res_params[full_name].append(this_params)

                                    res_human_seq_nums[full_name].append(human_sequence_number)
                                    res_sim_seq_nums[full_name].append(simulation_sequence_number)
                                
                                
                else: # not Sim Only


                    # loop through params that we we want to max over...
                    for d_model in hidden_sizes:

                        n_runs = 5

                        if part_name == 'Human_Only':
                            sim_lr = 0
                            these_human_lrs = human_lrs_train


                        else:

                            sim_lr = .001
                            these_human_lrs = human_lrs_finetune


                        for human_lr in these_human_lrs:

                            if model_name == 'Transformer':

                                for n_layers in transformer_layers:
                                    for n_head in transformer_attention_heads:

                                        # get the human loss...
                                        mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name,tsp, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = n_runs, which_loss = 'human_loss_results')
                                        full_name = "{}_{}_{}_{}".format(tsp, part_name, model_name, fu)

                                        res_losses[full_name].append(mean_loss)
                                        res_params[full_name].append(this_params)
                                        res_human_seq_nums[full_name].append(human_sequence_number)
                                        res_sim_seq_nums[full_name].append(simulation_sequence_number)

                            else: # not a transformer

                                n_head = 0
                                n_layers = 0   

                                # get the simulation loss...
                                mean_loss, sem_loss, train_sequence_number,simulation_sequence_number, human_sequence_number, this_params  = get_learning_curve(part_name,tsp, fu, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_runs = n_runs, which_loss = 'human_loss_results')
                                full_name = "{}_{}_{}_{}".format(tsp, part_name, model_name, fu)

                                res_losses[full_name].append(mean_loss)
                                res_params[full_name].append(this_params)
                                res_human_seq_nums[full_name].append(human_sequence_number)
                                res_sim_seq_nums[full_name].append(simulation_sequence_number)

                            
            

# save all results
best_params_seq = {}

for fu in fix_unit_types:
    for tsp in train_seq_parts:
        for part_name in part_names:
            for model_name in model_names:
                
                full_name = "{}_{}_{}_{}".format(tsp, part_name, model_name, fu)
                
                these_best_params = get_best_params(res_losses[full_name],res_params[full_name], res_sim_seq_nums[full_name], res_human_seq_nums[full_name])
                best_params_seq[full_name] = these_best_params

                if part_name == 'Simulated_Only':
                    full_name = "{}_{}_{}_{}".format(tsp, 'Simulated_Only_Pred_Human', model_name, fu)
                    these_best_params = get_best_params(res_losses[full_name],res_params[full_name], res_sim_seq_nums[full_name], res_human_seq_nums[full_name])
                    best_params_seq[full_name] = these_best_params
                    
# save
f = open(os.path.join(best_param_folder, "best_hyper_params.pkl"),"wb")

# write the python object (dict) to pickle file
pickle.dump(best_params_seq,f)

# close file
f.close()                    