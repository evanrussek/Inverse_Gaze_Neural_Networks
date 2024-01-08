#####################
##### Packages ######
#####################

import numpy as np
import matplotlib.pyplot as plt
import pickle
# import optuna
import time
import sys
import torch
import os
import argparse
import random

############################################
###### Load functions from other scripts 
############################################

from load_data_funs import load_data, gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only
from train_and_test_funs import test, train_on_simulation_then_human_with_intermediate_tests, test_record_each_output, compute_heldout_performance
from neural_nets import SimpleLSTM, SimpleMLP, SimpleGRU, SimpleTransformer, SimpleChoiceOnly

def main_as_fun(model_name: str ='LSTM', train_seq_part: str = 'fix_and_choice', n_simulation_sequences_train: int = 1e3, n_human_sequences_train: int = 0, n_sequences_test: int = 500, n_sequences_final_performance: int = 500, d_model: int = 128, n_layers: int = 2, n_head: int = 2, sim_lr: float = .001, human_lr: float = .001, batch_size: int = 32, dropout = 0, run_idx: int = 0, on_cluster: bool = True, test_batch_increment_sim: int = 200, test_batch_increment_human: int = 200, save_folder_name = 'Hyper_Param_Search', fix_unit = 'ID', save_file_name = ''):
        
        
    # add a 4th train_seq_part... choice_then_fix...
    
    print(locals())
    
    print(type(n_simulation_sequences_train))
    
    ####################################################
    #### Set cluster folders
    #############################################
    # set path to the human and simulation data
    if on_cluster:
        sim_data_path = '/scratch/gpfs/erussek/RNN_project/optimal_fixation_sims'
        human_data_path = '/scratch/gpfs/erussek/RNN_project/human_trials.json'
    else:
        sim_data_path = '/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'
        human_data_path = '/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/human_trials.json'

    # set up folder to save results
    if on_cluster:
        to_save_folder = '/scratch/gpfs/erussek/RNN_project/'+save_folder_name
    else:
        to_save_folder = '/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Code/RNNs/'+save_folder_name

    if not os.path.exists(to_save_folder):
        os.mkdir(to_save_folder)

    ###########################
    ### initialize results dictionary
    res_dict = {}

    ##################
    # set the random seed.
    random.seed(run_idx)
    torch.manual_seed(run_idx)

    # Functions to generate data
    gen_data_func_by_train_seq_part = {"fix_and_choice": gen_batch_data_fixations_choice, "fix_only": gen_batch_data_fixations_only, "choice_only": gen_batch_data_choice_only, "choice_then_fix": gen_batch_data_choice_only}
    gen_data_func_pre = gen_data_func_by_train_seq_part[train_seq_part]
    # set the fix unit
    gen_data_func = lambda a, b, c, use_human_data=False : gen_data_func_pre(a, b, c, fix_unit = fix_unit, use_human_data = use_human_data)
    
    # Number of input tokens depends on task and fix_unit
    
    if fix_unit == 'all':
        n_tokens_by_train_seq_part = {"fix_and_choice": 12, "fix_only": 9, "choice_only": 3, "choice_then_fix": 3}
    else:
        n_tokens_by_train_seq_part = {"fix_and_choice": 6, "fix_only": 3, "choice_only": 3, "choice_then_fix": 3}
        
    n_tokens = n_tokens_by_train_seq_part[train_seq_part]

    ######################
    ####### LOAD DATA ###
    #####################
    print("Loading Data")
    train_data_sim, val_data_sim, test_data_sim, train_data_human, val_data_human, test_data_human = load_data(sim_data_path, human_data_path,this_seed=run_idx,split_human_data=True)

    ##########################
    #### SETUP MODEL #########
    ##########################

    output_size = 3 # always return len 3 output size

    if model_name == 'MLP': # this should only be called w/ choice only setting
        model       = SimpleMLP(n_tokens, d_model, output_size)
    elif model_name == 'Transformer':    
        dim_feedforward = 4*d_model
        model = SimpleTransformer(n_tokens,d_model,dim_feedforward,output_size,nlayers = n_layers, nhead = n_head, dropout = dropout)
    elif model_name == 'GRU':
        model = SimpleGRU(n_tokens, d_model, output_size, dropout = dropout)
    elif model_name == 'LSTM':
        model = SimpleLSTM(n_tokens, d_model, output_size, dropout = dropout)
    elif model_name == 'Choice2P':
        model = SimpleChoiceOnly()
    else:
        exit("Invalid model name entered")


    # print(model)

    ##################################
    ### OTHER TRAINING PARAMETERS ####
    ##################################

    # non neural net training parameters
    criterion   = torch.nn.MSELoss()
    start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    #######################################################
    ##### Train the model and record learning curves #####
    ######################################################

    print("Training the model")
    simulation_loss_results, human_loss_results, train_sequence_number,human_sequence_number, simulation_sequence_number, model = train_on_simulation_then_human_with_intermediate_tests(model,train_data_sim, train_data_human,val_data_sim,val_data_human,criterion,device,batch_size,n_simulation_sequences_train, n_human_sequences_train, n_sequences_test, gen_data_func, sim_lr = sim_lr, human_lr = human_lr, test_batch_increment_sim=test_batch_increment_sim, test_batch_increment_human=test_batch_increment_human)

    # save results of this
    res_dict["simulation_loss_results"] =  simulation_loss_results
    res_dict["human_loss_results"] =  human_loss_results
    res_dict["train_sequence_number"] =  train_sequence_number
    res_dict["human_sequence_number"] =  human_sequence_number
    res_dict["simulation_sequence_number"] =  simulation_sequence_number

    ############################################################################################
    #### Evaluate model performance ##
    ############################################################################################

    print("Evaluating trained model performance")
    if train_seq_part != "choice_only":
        
        if train_seq_part == "choice_then_fix":
            gen_data_func = gen_batch_data_fixations_only

        n_back_vals = np.arange(1,20)

        r_sim_by_n_back = np.zeros(len(n_back_vals))
        r_human_by_n_back = np.zeros(len(n_back_vals))

        pct_correct_max_sim_by_n_back = np.zeros(len(n_back_vals))
        pct_correct_max_human_by_n_back = np.zeros(len(n_back_vals))

        pct_correct_min_sim_by_n_back = np.zeros(len(n_back_vals))
        pct_correct_min_human_by_n_back = np.zeros(len(n_back_vals))

        pct_correct_order_sim_by_n_back = np.zeros(len(n_back_vals))
        pct_correct_order_human_by_n_back = np.zeros(len(n_back_vals))
        
        mse_human_by_n_back = np.zeros(len(n_back_vals))
        n_items_human_by_n_back = np.zeros(len(n_back_vals))
        
        mse_sim_by_n_back = np.zeros(len(n_back_vals))
        n_items_sim_by_n_back = np.zeros(len(n_back_vals))

        for nb_idx, nb in enumerate(n_back_vals):

            r_sim_by_n_back[nb_idx], pct_correct_max_sim_by_n_back[nb_idx], pct_correct_min_sim_by_n_back[nb_idx], pct_correct_order_sim_by_n_back[nb_idx], mse_sim_by_n_back[nb_idx], n_items_sim_by_n_back[nb_idx] = compute_heldout_performance(model, test_data_sim, device, batch_size, n_sequences_final_performance,gen_data_func, nb, use_human_data=False)

            # print('Human data length: {}'.format(len(test_data_human)))
            
            r_human_by_n_back[nb_idx], pct_correct_max_human_by_n_back[nb_idx], pct_correct_min_human_by_n_back[nb_idx], pct_correct_order_human_by_n_back[nb_idx],  mse_human_by_n_back[nb_idx], n_items_human_by_n_back[nb_idx] = compute_heldout_performance(model, test_data_human, device, batch_size, n_sequences_final_performance,gen_data_func, nb, use_human_data=True)

    else:

        r_sim_by_n_back, pct_correct_max_sim_by_n_back, pct_correct_min_sim_by_n_back, pct_correct_order_sim_by_n_back, mse_sim_by_n_back, n_items_sim_by_n_back = compute_heldout_performance(model, test_data_sim, device, batch_size, n_sequences_final_performance,gen_data_func, 0, choice_only=True, use_human_data=False)

        r_human_by_n_back, pct_correct_max_human_by_n_back, pct_correct_min_human_by_n_back, pct_correct_order_human_by_n_back, mse_human_by_n_back, n_items_human_by_n_back = compute_heldout_performance(model, test_data_human, device, batch_size, n_sequences_final_performance, gen_data_func, 0, choice_only=True, use_human_data=True)


    # store these results
    res_dict["r_sim_by_n_back"] = r_sim_by_n_back
    res_dict["pct_correct_max_sim_by_n_back"] = pct_correct_max_sim_by_n_back
    res_dict["pct_correct_min_sim_by_n_back"] = pct_correct_min_sim_by_n_back
    res_dict["pct_correct_order_sim_by_n_back"] = pct_correct_order_sim_by_n_back

    res_dict["r_human_by_n_back"] = r_human_by_n_back
    res_dict["pct_correct_max_human_by_n_back"] = pct_correct_max_human_by_n_back
    res_dict["pct_correct_min_human_by_n_back"] = pct_correct_min_human_by_n_back
    res_dict["pct_correct_order_human_by_n_back"] = pct_correct_order_human_by_n_back
    
    res_dict["mse_human_by_n_back"] = mse_human_by_n_back
    res_dict["n_items_human_by_n_back"] = n_items_human_by_n_back
    res_dict["mse_sim_by_n_back"] = mse_human_by_n_back
    res_dict["n_items_sim_by_n_back"] = n_items_human_by_n_back


    ########################
    ### SAVE RESULTS #######
    ########################

    print("Saving results")

    # Don't save the model because it takes up too much space... 
    ## save the model with torch
    # model_file_name = save_file_name+'.pt'
    # model_full_file_name = os.path.join(to_save_folder, model_file_name)
    # torch.save(model, model_full_file_name)

    # save the results dict with np.save 
    res_file_name = save_file_name+'.pickle'
    res_full_file_name = os.path.join(to_save_folder, res_file_name)

    # save results file
    with open(res_full_file_name, 'wb') as f:    
        pickle.dump(res_dict, f) 

