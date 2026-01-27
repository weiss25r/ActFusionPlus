import json

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

    def get_params(self):
        return self.params