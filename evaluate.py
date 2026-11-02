"""Evaluate a fixed checkpoint/threshold; no threshold search or test selection."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS
import shm  # noqa: F401
from shm.metrics import confusion_counts, foreground_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config'); parser.add_argument('checkpoint')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError('Threshold must be in [0, 1]')
    cfg = Config.fromfile(args.config)
    if os.environ.get('CD_DATA_ROOT'):
        cfg[f'{args.split}_dataloader'].dataset.data_root = os.environ['CD_DATA_ROOT']
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    cfg.model.pretrained = None
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    loader_cfg = cfg[f'{args.split}_dataloader'].copy()
    loader_cfg.batch_size = 1; loader_cfg.num_workers = 0; loader_cfg.persistent_workers = False
    loader = Runner.build_dataloader(loader_cfg)
    counts = np.zeros(4, dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            inputs = torch.stack(batch['inputs']).to(args.device)
            logits = model.predict(inputs, batch['data_samples'], mode='tensor')
            probability = logits.float().softmax(dim=1)[0, 1]
            if not torch.isfinite(probability).all():
                raise FloatingPointError('Non-finite model probabilities')
            target = batch['data_samples'][0].gt_sem_seg.data.cpu().numpy()
            counts += confusion_counts((probability >= args.threshold).cpu().numpy(), target)
    report = dict(split=args.split, threshold=args.threshold, images=len(loader.dataset), **foreground_metrics(counts))
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
