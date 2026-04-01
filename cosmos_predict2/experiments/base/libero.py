# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

# ── Changed: new import paths for Predict2.5 ──────────────────────────────────
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)

# ── Changed: checkpoint handled via MODEL_CHECKPOINTS registry ────────────────
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey, ModelSize

DEFAULT_CHECKPOINT_2B = MODEL_CHECKPOINTS[ModelKey(post_trained=True)]

cs = ConfigStore.instance()

# ── Dataset ───────────────────────────────────────────────────────────────────
example_video_dataset_libero = L(VideoDataset)(
    dataset_dir="/net/data/llmpoison/libero/cosmos25_libero",
    num_frames=121,
    video_size=(432, 432),
)

# ── Changed: get_generic_dataloader instead of raw DataLoader ─────────────────
dataloader_train_libero = L(get_generic_dataloader)(
    dataset=example_video_dataset_libero,
    sampler=L(get_sampler)(dataset=example_video_dataset_libero),
    batch_size=2,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# ── Experiment ────────────────────────────────────────────────────────────────
# EXP=predict2_video2world_training_2b_libero_480 CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 --master_port=12341   -m scripts.train   --config=cosmos_predict2/_src/predict2/configs/video2world/config.py   -- experiment=${EXP}
predict2_video2world_training_2b_libero_480 = dict(
    defaults=[
        # ── Changed: inherits from the registered base checkpoint experiment ──
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    # ── Changed: dataloader passed directly, not via cs.store override ────────
    dataloader_train=dataloader_train_libero,
    checkpoint=dict(
        save_iter=500,
        load_path=DEFAULT_CHECKPOINT_2B.s3.uri,   # now points at post-trained weights
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_libero_480",
    ),
    # ── Changed: updated LR and weight_decay to match Predict2.5 GR1 values ───
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    # ── Changed: updated scheduler f_max/f_min to Predict2.5 values ──────────
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=2000,
        straggler_detection=dict(
            enabled=False,
        ),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=500, save_s3=False),
            every_n_sample_ema=dict(every_n=500, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
)

# ── Registration ──────────────────────────────────────────────────────────────
for _item in [
    predict2_video2world_training_2b_libero_480,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )