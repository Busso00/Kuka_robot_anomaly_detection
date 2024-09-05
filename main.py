'''
Project: MLinApps-Proj
Author:MLinApps-Proj group 2024/08
Members:
Haochen He S307771
Federico Bussolino S317641
Marco D'almo  S301199
Youness Bouchari S323624
'''

import logging
import wandb
from parse_args import parse_arguments
from load_data import TADGANLOADER,MSCREDLOADER,VAELSTMLOADER, MTGFLOWLOADER_action
from experiments.MSCRED import MSCREDExperiment
from experiments.TADGAN import TADGANExperiment
from experiments.VAELSTM import VAELSTMExperiment
#from experiments.VAELSTM2 import VAELSTM2Experiment
from experiments.MTGFLOW import MTGFLOWExperiment
import tensorflow as tf
from datetime import datetime
from metrics import compute_average_score,metrics_by_point_vectorized, convert
import numpy as np

def setup_experiment(opt, config=None):

    if opt['experiment'] == 'VAELSTM':
        experiment = VAELSTMExperiment(opt)
        train_data, collision_data, collision_time = VAELSTMLOADER(opt)

       
        return experiment, train_data, collision_data, collision_time
    
    if opt['experiment'] == 'VAELSTM2':
        experiment = VAELSTM2Experiment(opt, config)
        train_data, collision_data, collision_times_validation_init, collision_time, collision_times_validation_end = VAELSTM2LOADER(opt)

       
        return experiment, train_data, collision_data, collision_times_validation_init, collision_time, collision_times_validation_end
    
    elif opt['experiment'] == 'TADGAN':
        experiment = TADGANExperiment(opt)
        train_data, collision_data, collision_times_validation_init, collision_time, collision_times_validation_end = TADGANLOADER(
            opt)

        return experiment, train_data, collision_data, collision_times_validation_init, collision_time, collision_times_validation_end

    elif opt['experiment'] == 'MTGFLOW':
        #run a series of experiments

        experiments = []
        train_datas = []
        collision_datas = []
        eval_times = []
        actions_collisions = []
        actions_trains = []

        n = 0
        for window_size in opt['possible_window_size']:
          for num_features in opt['possible_num_features']:
            opt['window_size'] = window_size #set before load
            opt['num_features'] = num_features
            train_data, actions_train, collision_data, eval_time_df, actions_collision = MTGFLOWLOADER_action(opt) #loader called only at different window sizes
            
            train_datas.append(train_data)
            collision_datas.append(collision_data)
            eval_times.append(eval_time_df)
            actions_collisions.append(actions_collision)
            actions_trains.append(actions_train)

            #set espleriment config as one over the possible choices (crossvalidate)
            for hidden_size in opt['possible_hidden_size']:
              for n_blocks in opt['possible_n_blocks']:
                for dropout in opt['possible_dropout']:
                  for lr in opt['possible_lr']:
                        
                      opt['hidden_size'] = hidden_size
                      opt['dropout'] = dropout
                      opt['lr'] = lr
                      opt['n_blocks'] = n_blocks
                      
                      exp = MTGFLOWExperiment(opt.copy(), n)
                      experiments.append(exp)
                      n += 1

        return experiments, train_datas, actions_trains, collision_datas, eval_times, actions_collisions


    
    
    elif opt['experiment'] == 'MSCRED':

        train_windows, test_windows, time_windows = MSCREDLOADER(opt)
        experiment = MSCREDExperiment(opt,train_windows,test_windows)
        return experiment, time_windows

def main(opt):

    if opt['experiment'] == 'TADGAN':
        print("TADGAN experiment")
        opt['window_size'] = 100
        opt['train'] = True
        opt['feature_num'] = 51
        opt['action'] = True
        opt['feature_analysis'] = False
        opt['point-based'] = False

        if not opt['action']:
            opt['feature_num'] = 50
            if opt['feature_analysis']:
                opt['feature_num'] = 32
        else:
            opt['feature_num'] = 51
            if opt['feature_analysis']:
                opt['feature_num'] = 33

        if opt['point-based']:
            opt['window_size'] = 100

        experiment, train_data, collision_data, collision_times_validation_init, collision_time, collision_times_validation_end = setup_experiment(
            opt)
        experiment.fit(train_data)

        anomalies1, anomalies2, reconstruction_errors = experiment.detect_anomalies(collision_data, None,
                                                                                    collision_times_validation_init,
                                                                                    collision_times_validation_end,
                                                                                    collision_time)
        

       

    elif opt['experiment'] == 'MTGFLOW':

        #additional parameters, can run multiple experiments
        opt['possible_window_size'] = [100] 
        opt['stride'] = 30
        opt['possible_num_features'] = [51] 
        opt['possible_hidden_size'] = [47] #Hidden layer size for MADE (and each MADE block in an MAF) #16 for 50 features, 
        opt['possible_n_blocks'] = [1] #Number of blocks to stack in a model (MADE in MAF) 
        opt['batch_size'] = 64
        opt['possible_lr'] = [5e-5] # 2e-4 with step scheduler and no overlapping, 5e-5 with step scheduler and overlapping
        opt['possible_dropout'] = [0.0]
        opt['epochs'] = 500
        opt['tf'] = True

        #best so far 47, 50, 2e-4, 0.0
        
        #best now: n_blcks=3, hidden_size=16, window_size=100, lr=1e-4, wd=5e-3, dropout=0.
        print("MTGFLOW experiments")
        experiments, train_datas, actions_trains, collision_datas, collision_times, actions_collisions = setup_experiment(opt.copy())

        for n,experiment in enumerate(experiments):
            print(f"experiment {n}/{len(experiments)}")
            print("config:")
            print(experiment.opt)

            index = experiment.opt['possible_window_size'].index(experiment.opt['window_size'])*len(experiment.opt['possible_num_features'])
            index += experiment.opt['possible_num_features'].index(experiment.opt['num_features'])
            
            train_data = train_datas[index]
            collision_data = collision_datas[index]
            collision_time = collision_times[index]
            
            wandb.login(key="<YOUR API KEY>")
                
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # wandb initialization
            wandb.init(
                project="FP01",
                tags=["MTGFLOW", experiment.opt['experiment']],
                name=f"{opt['experiment']}_{time}",
                config=experiment.opt
            )

            print("--------------------------START TRAINING--------------------------")
            experiment.train(train_data, time)
            print("----------------------------START TEST----------------------------")

            experiment.detect_anomalies(collision_data, collision_time)


    if opt['experiment'] == 'MSCRED':
        print("MSCRED experiment")
        opt['window_size'] = 100
        opt['overlap_train'] = 0
        opt['batch_size'] = 4
        opt['learning_rate'] = 0.0005
        opt['training_iters']= 150
        opt['step_max'] = 5
        experiment, time_windows= setup_experiment(opt)
        time_windows =  convert(time_windows[opt['step_max']-1:])

        experiment.model.compile()
        experiment.model.train()
        r = experiment.model.test()
        scores = experiment.score(r,opt)
        scores = compute_average_score(scores, opt)
        recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices = experiment.evaluate(scores, time_windows)


if __name__ == '__main__':
    opt = parse_arguments()
    opt['experiment'] = "MSCRED"

    
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3096)])
        except RuntimeError as e:
            print(e)
            
    if opt['experiment'] == 'VAELSTM':
        print("VAELSTM experiment")
        experiment, train_data, collision_data, collision_time = setup_experiment(opt.copy())
        if not opt['test']:
            experiment.fit(train_data)
        else:
            experiment.load_model()
        experiment.detect_anomalies(collision_data,collision_time)

main(opt)
