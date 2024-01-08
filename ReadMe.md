This folder contains code accompanying the submission "Inverting cognitive models with neural networks to infer preferences from fixations"

Code was made with reference to NYU's deep learning course shared code, https://github.com/Atcold/NYU-DLSP21. The file 'sequential tasks' is taken from code for that course. Scripts from that course were also referenced for building neural networks in pytorch.

The manuscript covers 4 separate training and testing tasks:

Training on simulated data and testing on simulated data

Training on simulated data and testing on human data

Training on human data alone and testing on human data

Training on simulated and human data and testing on human data

For each of these tasks, we train three separate sequential models: LSTMs, GRUs and Transformers, which map sequences of fixations to predictions of utilities.

We also train a set of control models which use MLPs to map hand-crafted feautures to predictions of utilities.

For each of these tasks, for each model, we first do a grid-search through hyper-parameters to find the hyperparameters that provide the best performance on held-out validation data. The search is done in 'Search_Hyper_Params_Sequential.py' and 'Search_Hyper_Params_Control.py'. These scripts were run on clusters of GPUs (example submission script in script 'RNN_gpu_parallel.slurm').

Following a search, the best parameters for each model/task are identified and saved with the scripts 'ID_Best_Hyperparams_Sequential.py' and 'ID_Best_Hyperparams_Control.py'.

Once the best hyperparameters are identified, these are used to train models in the scripts 'Train_at_Best_Params_Sequential.py' and 'Train_at_Best_Params_Control.py'. 

Because the choice-only model only has 2 parameters, identifying best hyperparameters is not relevant. This model is trained in 'TrainChoice2P.py'.

Given the results of these, figures and analysis are made in the notebook 'MakeFigures.ipynb'

The following scripts are called in training: 

neural_nets.py defines the networks
load_data_funs.py structures data as input for training and testing for differnet tasks
sequential_tasks.py is from the nyu deep learning course and has functions used in load_data_funs.py
main_as_fun.py trains and tests a network, with settings specified in input parameters
