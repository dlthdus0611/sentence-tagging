import os
import torch
import random
import pickle
import neptune
import argparse
import numpy as np
import config as config
from pathlib import Path
from trainer import Trainer
from helper.configure import Configure

def main():
    parser = argparse.ArgumentParser(description='Settings for SEN-CLS')
    parser.add_argument('--exp_name', default='Base', type=str)
    parser.add_argument('--exp_num', default='1', type=str)
    parser.add_argument('--tag', default='Default', type=str, help='tag')
    parser.add_argument('--config_path', default='config/0settings.json', type=str)
    parser.add_argument('--output_dir', type=str, default='results/', help='Model path')
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--server', type=str, default='lyceum_4')
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--test_path', type=str, default='results/73', help='Model path')    
    ### Scheduler
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--warm_epoch', type=int, default=3)
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--tmax', type=int, default=10)
    ### Training Parameter
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--initial_lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    args = parser.parse_args()
    configs = Configure(config_json_file=args.config_path)

    # Random Seed
    seed = configs.train.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # neptune setting
    if args.logging:
        api = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OTRmMzMyMC0wNDA0LTRlZDAtYTg1Ni0zZTU3NDg3NGQ3YTYifQ=="
        neptune.init("dlthdus8450/sen-cls", api_token=api)
        temp = neptune.create_experiment(name=args.exp_num, params=vars(args))
        experiment_num = str(temp).split('-')[-1][:-1] 
        neptune.append_tag(args.tag)
    else:
        experiment_num = 'tmp2'

    output_path = Path(args.output_dir, experiment_num)
    output_path.mkdir(parents=True, exist_ok=True)

    info = {
            "output_dir": output_path,
            'test_path': args.test_path,
            }

    trainer = Trainer(args, configs, info)

    if not args.do_test:
        trainer.train()
    elif args.do_test:
        trainer.test()

if __name__ == '__main__':
    main()