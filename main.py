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

    args = parser.parse_args()

    config = ActFusionConfig(config_file=args.config)
    all_params = config.get_params()

    # Add pos, n_mask, and patch_size from config to args
    args.pos = all_params.get('pos', 'none')
    args.n_mask = all_params.get('n_mask', 10)
    args.patch_size = all_params.get('patch_size', 10)

    naming = args.result_dir
    device = torch.device('cuda')

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    print(device)
    if all_params['dataset_name'] == '50salads':
        wandb.init(project='50s_diffusion_integrate_++')
    elif all_params['dataset_name'] == 'gtea':
        wandb.init(project='gtea_diffusion_integrate_++')
    else:
        wandb.init(project='bf_diffusion_integrate_++')

    wandb.run.name = args.result_dir
    wandb.config.update(vars(args), allow_val_change=True)
    wandb.config.update(all_params, allow_val_change=True)

    feature_dir = os.path.join(all_params['root_data_dir'], all_params['dataset_name'], 'features')
    label_dir = os.path.join(all_params['root_data_dir'], all_params['dataset_name'], 'groundTruth')
    mapping_file = os.path.join(all_params['root_data_dir'], all_params['dataset_name'], 'mapping.txt')
    print("mapping_file: ", mapping_file)
    actions_dict = read_mapping_dict(mapping_file)

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)
    split = args.split
    print("split: ",split)

    train_video_list = np.loadtxt(os.path.join(
        all_params['root_data_dir'], all_params['dataset_name'], 'splits', f'train.split{split}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        all_params['root_data_dir'], all_params['dataset_name'], 'splits', f'test.split{split}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    test_preprocessor_params = {
            'feature_dir':feature_dir,
            'label_dir':label_dir,
            'video_list':test_video_list,
            'event_list':event_list,
            'sample_rate':all_params['sample_rate'],
            'temporal_aug':all_params['temporal_aug'],
            'boundary_smooth':all_params['boundary_smooth']
        }
    
    if not args.test:
        train_preprocessor_params = {
            'feature_dir':feature_dir,
            'label_dir':label_dir,
            'video_list':train_video_list,
            'event_list':event_list,
            'sample_rate':all_params['sample_rate'],
            'temporal_aug':all_params['temporal_aug'],
            'boundary_smooth':all_params['boundary_smooth']
        }
        train_train_dataset = VideoFeatureDataset(train_preprocessor_params, num_classes, mode='train')
        train_test_dataset = VideoFeatureDataset(train_preprocessor_params, num_classes, mode='test')

    dataset_name = all_params['dataset_name']

    test_test_dataset = VideoFeatureDataset(test_preprocessor_params, num_classes, mode='test')

    trainer = Trainer(dict(all_params['encoder_params']), dict(all_params['decoder_params']), dict(all_params['diffusion_params']),
        event_list, all_params['sample_rate'], all_params['temporal_aug'], 
        all_params['set_sampling_seed'], all_params['postprocess'],
        device=device, args=args
    )

    if not os.path.exists(all_params['result_dir']):
        os.makedirs(all_params['result_dir'])

    if args.test:
        device = torch.device('cuda')
        mode = 'decoder-agg'

        if args.ckpt:
            model_path = os.path.join('ckpt', dataset_name, 'split'+str(args.split)+'.model')
        else:
            model_path = os.path.join('result', args.result_dir, dataset_name, 'split'+str(args.split), 'best_combined_model.pth')
        print("model loaded:", model_path)
        result_path = os.path.join('result', args.result_dir, dataset_name, 'split'+str(args.split))

        # For test mode, always run both TAS and LTA inference
        """
        print("TAS inference")
        test_result_dict = trainer.test(
            test_test_dataset, mode, device, label_dir,
            result_dir=result_path, model_path=model_path, args=args, all_params=all_params, obs_p=1.0)
        """
        print("LTA inference")
        obs_ps = [0.2, 0.3]
        for obs_p in obs_ps:
            print("LTA inference: obs_p", obs_p)
            test_result_dict = trainer.test(
                test_test_dataset, mode, device, label_dir, args=args,
                result_dir=result_path, model_path=model_path, all_params=all_params, obs_p=obs_p)
    else:
        trainer.train(train_train_dataset, train_test_dataset, test_test_dataset,
            all_params['loss_weights'], all_params['class_weighting'], all_params['soft_label'],
            all_params['num_epochs'], all_params['batch_size'], all_params['learning_rate'], all_params['weight_decay'],
            label_dir=label_dir, result_dir=os.path.join('result', naming, dataset_name,'split'+str(split)),
            log_freq=all_params['log_freq'], 
            log_train_results=all_params['log_train_results'], args=args, all_params=all_params
        )
