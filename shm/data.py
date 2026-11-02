"""Paired RGB images and explicit binary mask loading; no GDAL dependency."""
from pathlib import Path
import numpy as np
from PIL import Image
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class CDPairDataset(BaseSegDataset):
    METAINFO = dict(classes=('background', 'changed'), palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

    def load_data_list(self):
        records = super().load_data_list()
        second = Path(self.data_prefix['img_path2'])
        for record in records:
            record['img_path2'] = str(second / Path(record['img_path']).name)
            if not Path(record['img_path2']).is_file():
                raise FileNotFoundError(record['img_path2'])
        return records


@TRANSFORMS.register_module()
class LoadCDPair(BaseTransform):
    def transform(self, results):
        with Image.open(results['img_path']) as image:
            first = np.asarray(image.convert('RGB'))
        with Image.open(results['img_path2']) as image:
            second = np.asarray(image.convert('RGB'))
        if first.shape != second.shape:
            raise ValueError('The two images must have identical spatial dimensions')
        with Image.open(results['seg_map_path']) as image:
            raw = np.asarray(image)
        if raw.ndim == 3:
            raw = raw.max(axis=2)
        if raw.shape != first.shape[:2]:
            raise ValueError('Mask and image dimensions differ')
        # Explicitly support {0,1} and {0,255}. Gray values use a fixed threshold.
        threshold = 0.5 if raw.max(initial=0) <= 1 else 128
        results['gt_seg_map'] = (raw >= threshold).astype(np.uint8)
        results['img'] = np.concatenate([first, second], axis=2)
        results['img_shape'] = results['ori_shape'] = first.shape[:2]
        fields = results.setdefault('seg_fields', [])
        if 'gt_seg_map' not in fields:
            fields.append('gt_seg_map')
        return results
