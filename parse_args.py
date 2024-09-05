''''
Project: MLinApps-Proj
Author:MLinApps-Proj group 2024/08
Members:
Haochen He S307771
Federico Bussolino S317641
Marco D'almo  S301199
Youness Bouchari S323624
'''


import argparse
import tensorflow as tf


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default='TADGAN',
                        choices=['MSCRED', 'VAELSTM', 'MTGFLOW','TADGAN'])

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--max_iterations', type=int, default=5000, help='Number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=64,help='Batch size.')
    parser.add_argument('--window_size', type=int, default=100, help='Window size.')
    parser.add_argument('--time_step', type=int, default=1, help='Time step.')
    parser.add_argument('--x_dim', type=int, default=50, help='Feature number.')
    parser.add_argument('--lstm_h_dim', type=int, default=25, help='Lstm hidden dimensions.')
    parser.add_argument('--z_dim', type=int, default=25, help='Z dimensions.')
    parser.add_argument('--epoch_num', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--threshold', type=float, default=0.03, help='Threshold.')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='Epsilon value.')
    parser.add_argument('--gradient_clipvalue', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('--scaler_value', type=float, default=1, help='Value of scaler.')
    parser.add_argument('--feature_num', type=int, default=50, help='Number of features.')
    parser.add_argument('--output_path', type=str, default='.',
                        help='Where to create the output directory containing logs and weights.')
    parser.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU.')
    parser.add_argument('--test', action='store_true', help='If set, the experiment will skip training.')
    parser.add_argument('--train', action='store_true', help='If set,  will do training.')
    parser.add_argument('--action', action='store_true', help='If set,  will use actions.')

    # Build options dict
    opt = vars(parser.parse_args())

    #if not opt['cpu']:
    #    physical_devices = tf.config.list_physical_devices('GPU')
    #    if not physical_devices:
    #        raise AssertionError('You need a CUDA capable device in order to run this experiment. See `--cpu` flag.')
    #    else:
    #        try:  
    #            # Currently, memory growth needs to be the same across GPUs
    #            for device in physical_devices:
    #                tf.config.experimental.set_memory_growth(device, True)
    #                print('ok')
    #        except:
    #            raise AssertionError('Invalid device or cannot modify virtual devices once initialized.')

    return opt
