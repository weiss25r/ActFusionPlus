"""
This code is built upon DiffAct: https://github.com/Finspire13/DiffAct
"""
import os
import torch
import argparse
import numpy as np

from src.dataset import VideoFeatureDataset
from src.utils import read_mapping_dict
from src.trainer import Trainer
from src.config import ActFusionConfig

import wandb
import random
from torch.backends import cudnn

# Seed fix 
#seed = 13452
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#cudnn.benchmark, cudnn.deterministic = False, True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--test', action='store_true', help='only test mode')
    parser.add_argument('--result_dir', type=str, default='actfusion')
    parser.add_argument('--ckpt', action='store_true', help='inference with checkpoint models')

    user_args = parser.parse_args()

    config = ActFusionConfig(config_file=user_args.config)
    model_params = config.params

    # Add pos, n_mask, and patch_size from config to args
    user_args.pos = model_params.get('pos', 'none')
    user_args.n_mask = model_params.get('n_mask', 10)
    user_args.patch_size = model_params.get('patch_size', 10)

    naming = user_args.result_dir
    device = torch.device('cuda')

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    print(device)

    if model_params['dataset_name'] == '50salads':
        wandb.init(project='50s_diffusion_integrate_++')
    elif model_params['dataset_name'] == 'gtea':
        wandb.init(project='gtea_diffusion_integrate_++')
    else:
        wandb.init(project='bf_diffusion_integrate_++')

    wandb.run.name = user_args.result_dir
    wandb.config.update(vars(user_args), allow_val_change=True)
    wandb.config.update(model_params, allow_val_change=True)

    feature_dir = os.path.join(model_params['root_data_dir'], model_params['dataset_name'], 'features')
    label_dir = os.path.join(model_params['root_data_dir'], model_params['dataset_name'], 'groundTruth')
    mapping_file = os.path.join(model_params['root_data_dir'], model_params['dataset_name'], 'mapping.txt')
    print("mapping_file: ", mapping_file)
    actions_dict = read_mapping_dict(mapping_file)

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)
    split = user_args.split
    print("split: ",split)

    train_video_list = np.loadtxt(os.path.join(
        model_params['root_data_dir'], model_params['dataset_name'], 'splits', f'train.split{split}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        model_params['root_data_dir'], model_params['dataset_name'], 'splits', f'test.split{split}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    test_preprocessor_params = {
            'feature_dir':feature_dir,
            'label_dir':label_dir,
            'video_list':test_video_list,
            'event_list':event_list,
            'sample_rate':model_params['sample_rate'],
            'temporal_aug':model_params['temporal_aug'],
            'boundary_smooth':model_params['boundary_smooth']
        }
    
    if not user_args.test:
        train_preprocessor_params = {
            'feature_dir':feature_dir,
            'label_dir':label_dir,
            'video_list':train_video_list,
            'event_list':event_list,
            'sample_rate':model_params['sample_rate'],
            'temporal_aug':model_params['temporal_aug'],
            'boundary_smooth':model_params['boundary_smooth']
        }
        train_train_dataset = VideoFeatureDataset(train_preprocessor_params, num_classes, mode='train')
        train_test_dataset = VideoFeatureDataset(train_preprocessor_params, num_classes, mode='test')

    dataset_name = model_params['dataset_name']

    test_test_dataset = VideoFeatureDataset(test_preprocessor_params, num_classes, mode='test')

    trainer = Trainer(model_params, device, event_list, user_args)

    if not os.path.exists(model_params['result_dir']):
        os.makedirs(model_params['result_dir'])

    if user_args.test:
        trainer.run_inference(test_test_dataset, device, label_dir)
        
    else:
        trainer.train(train_train_dataset, train_test_dataset, test_test_dataset,
            label_dir=label_dir, result_dir=os.path.join('result', naming, dataset_name,'split'+str(split))
        )
