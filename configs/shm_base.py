"""Portable Base/LEVIR-style example; entry points apply environment paths."""

default_scope = 'mmseg'
data_root = 'data/LEVIR-CD'
pretrained = None
work_dir = 'work_dirs/shm_base'
custom_imports = dict(imports=['shm'], allow_failed_imports=False)
model = dict(
    type='SiameseMambaVision',
    backbone_cfg=dict(dim=128, in_dim=64, depths=(3, 3, 10, 5),
                      window_size=(8, 8, 64, 32), num_heads=(2, 4, 8, 16),
                      mlp_ratio=4.0, drop_path_rate=0.4, layer_scale=1e-5,
                      out_indices=(0, 1, 2, 3)),
    neck_out_channels='auto_half', min_neck_channels=128, align_mode='pre',
    diff_module_type='gated', gate_type='difference', gate_channels='channel_spatial',
    gate_arrangement='parallel', use_diff_in_gate=True, use_product_feature=True,
    norm_cfg=dict(type='GN', num_groups=32), fpn_dim=256, num_classes=2,
    decoder_fuse_mode='concat', use_deep_supervision=True,
    aux_loss_weights=(0.1, 0.1, 0.1), use_edge_supervision=True,
    lambda_edge=0.2, edge_width=3, edge_pos_weight=5.0,
    class_weight_bg=1.0, class_weight_change=5.0, lambda_dice=0.5,
    pretrained=pretrained,
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)),
)
train_pipeline = [dict(type='LoadCDPair'),
                  dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
                  dict(type='RandomFlip', prob=0.5, direction='horizontal'), dict(type='PackSegInputs')]
eval_pipeline = [dict(type='LoadCDPair'), dict(type='PackSegInputs')]


def loader(split, training=False):
    return dict(batch_size=2 if training else 1, num_workers=4, persistent_workers=True,
                sampler=dict(type='InfiniteSampler' if training else 'DefaultSampler', shuffle=training),
                dataset=dict(type='CDPairDataset', data_root=data_root,
                             data_prefix=dict(img_path=f'{split}/t1', img_path2=f'{split}/t2', seg_map_path=f'{split}/label'),
                             reduce_zero_label=False, pipeline=train_pipeline if training else eval_pipeline))


train_dataloader = loader('train', True)
val_dataloader = loader('val')
test_dataloader = loader('test')
del loader
val_evaluator = dict(type='ChangeMetric')
test_evaluator = dict(type='ChangeMetric')
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(type='AmpOptimWrapper',
                     optimizer=dict(type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05),
                     paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1), bias=dict(decay_mult=0.0), norm=dict(decay_mult=0.0))),
                     clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
                   dict(type='PolyLR', eta_min=1e-8, power=1.0, by_epoch=False, begin=500, end=20000)]
default_hooks = dict(timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=100),
                     param_scheduler=dict(type='ParamSchedulerHook'), sampler_seed=dict(type='DistSamplerSeedHook'),
                     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=1,
                                     save_best='change/f1', rule='greater'))
log_processor = dict(by_epoch=False)
randomness = dict(seed=3407, deterministic=False)
env_cfg = dict(cudnn_benchmark=True, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))
launcher = 'none'
load_from = None
resume = False
