"""
This code is built upon DiffAct: https://github.com/Finspire13/DiffAct
"""
import argparse
import numpy as np

from src.pipeline import ActFusionPipeline

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

    print(user_args)

    pipeline = ActFusionPipeline(visible_devices="1")
    pipeline.run(
        user_args=user_args
    )
    