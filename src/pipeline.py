import json, torch, wandb, os
import numpy as np
from src.trainer import Trainer
from src.dataset import VideoFeatureDataset
from src.utils import read_mapping_dict

class ActFusionPipeline:
    def __init__(self, visible_devices):
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    def run(self, user_args, dirs, wandb_project_name=None):
        config = ActFusionConfig(config_file=user_args.config)
        model_params = config.params

        # Add pos, n_mask, and patch_size from config to args
        user_args.pos = model_params.get('pos', 'none')
        user_args.n_mask = model_params.get('n_mask', 10)
        user_args.patch_size = model_params.get('patch_size', 10)

        naming = user_args.result_dir
        device = torch.device('cuda')

        print(device)

        if wandb_project_name == None:
            if model_params['dataset_name'] == '50salads':
                wandb.init(project='50s_diffusion_integrate_++', reinit=True)
            elif model_params['dataset_name'] == 'gtea':
                wandb.init(project='gtea_diffusion_integrate_++', reinit=True)
            else:
                wandb.init(project='bf_diffusion_integrate_++', reinit=True)
        else:
            wandb.init(project=wandb_project_name, reinit=True)

        wandb.run.name = user_args.result_dir
        wandb.config.update(vars(user_args), allow_val_change=True)
        wandb.config.update(model_params, allow_val_change=True)

        if dirs == None:
            feature_dir = os.path.join(model_params['root_data_dir'], model_params['dataset_name'], 'features')
            label_dir = os.path.join(model_params['root_data_dir'], model_params['dataset_name'], 'groundTruth')
            mapping_file = os.path.join(model_params['root_data_dir'], model_params['dataset_name'], 'mapping.txt')
            print("mapping_file: ", mapping_file)
        else:
            feature_dir = dirs['feature_dir']
            label_dir = dirs['label_dir']
            mapping_file = dirs['mapping_file']
        
        actions_dict = read_mapping_dict(mapping_file)

        event_list = np.loadtxt(mapping_file, dtype=str)
        event_list = [i[1] for i in event_list]
        num_classes = len(event_list)
        split = user_args.split
        print("split: ",split)

        if dirs == None:
            train_video_list = np.loadtxt(os.path.join(
                model_params['root_data_dir'], model_params['dataset_name'], 'splits', f'train.split{split}.bundle'), dtype=str)
            test_video_list = np.loadtxt(os.path.join(
                model_params['root_data_dir'], model_params['dataset_name'], 'splits', f'test.split{split}.bundle'), dtype=str)
        else:
            train_video_list = np.loadtxt(dirs['train_split'], dtype=str)
            test_video_list = np.loadtxt(dirs['test_split'], dtype=str)
            val_video_list = np.loadtxt(dirs['val_split'], dtype=str)

        train_video_list = [i.split('.')[0] for i in train_video_list]
        val_video_list = [i.split('.')[0] for i in val_video_list]
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

            val_preprocessor_params = {
                'feature_dir':feature_dir,
                'label_dir':label_dir,
                'video_list':val_video_list,
                'event_list':event_list,
                'sample_rate':model_params['sample_rate'],
                'temporal_aug':model_params['temporal_aug'],
                'boundary_smooth':model_params['boundary_smooth']
            }
            

            train_dataset = VideoFeatureDataset(train_preprocessor_params, num_classes, mode='train')
            val_dataset = VideoFeatureDataset(val_preprocessor_params, num_classes, mode='test')

        dataset_name = model_params['dataset_name']

        test_dataset = VideoFeatureDataset(test_preprocessor_params, num_classes, mode='test')

        trainer = Trainer(model_params, device, event_list, user_args, mapping_file)

        if not os.path.exists(model_params['result_dir']):
            os.makedirs(model_params['result_dir'])

        if user_args.test:
            trainer.run_inference(test_dataset, device, label_dir)
            
        else:
            trainer.train(train_dataset, val_dataset,
                label_dir=label_dir, result_dir=os.path.join('result', naming, dataset_name,'split'+str(split))
            )

        wandb.finish()

class ActFusionConfig:
    def __init__(self, config_file: str):
        self.__load_config_file(config_file)

    def __load_config_file(self, config_file):
        all_params = json.load(open(config_file))

        if 'result_dir' not in all_params:
            all_params['result_dir'] = 'result'

        if 'log_train_results' not in all_params:
            all_params['log_train_results'] = True

        if 'soft_label' not in all_params:
            all_params['soft_label'] = None

        if 'postprocess' not in all_params:
            all_params['postprocess'] = {
                'type': None,
                'value': None
            }

        if 'use_instance_norm' not in all_params['encoder_params']:
            all_params['encoder_params']['use_instance_norm'] = False

        if 'detach_decoder' not in all_params['diffusion_params']:
            all_params['diffusion_params']['detach_decoder'] = False

        if 'pos' not in all_params:
            all_params['pos'] = 'none'

        if 'n_mask' not in all_params:
            all_params['n_mask'] = 10

        if 'patch_size' not in all_params:
            all_params['patch_size'] = 10

        assert all_params['loss_weights']['encoder_boundary_loss'] == 0

        self.params = all_params