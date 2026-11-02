"""Train the portable SHM configuration without hidden experiment presets."""
import argparse
import os
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import shm  # noqa: F401: register model, data transforms and metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if os.environ.get('CD_DATA_ROOT'):
        for split in ['train', 'val', 'test']:
            cfg[f'{split}_dataloader'].dataset.data_root = os.environ['CD_DATA_ROOT']
    if os.environ.get('MAMBAVISION_PRETRAINED'):
        cfg.model.pretrained = os.environ['MAMBAVISION_PRETRAINED']
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.resume = args.resume
    if cfg.model.get('pretrained') is None and not cfg.resume and not cfg.get('load_from'):
        raise ValueError('Set MAMBAVISION_PRETRAINED or model.pretrained before training')
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
