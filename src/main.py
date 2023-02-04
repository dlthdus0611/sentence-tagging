import os
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from trainer import Trainer
from utils.configure import Configure

def main():
    parser = argparse.ArgumentParser(description='Settings for SEN-CLS')
    parser.add_argument('--config_path', default='config.json', type=str)
    parser.add_argument('--exp_num', default='1', type=str)
    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)
    
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

    save_path = Path(configs.save_path, args.exp_num)
    save_path.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(args, configs, save_path)

    if args.do_train:
        trainer.train()
    elif args.do_test:
        trainer.test()
    elif args.do_predict:
        trainer.predict()

if __name__ == '__main__':
    main()