"""
Copyright 2022 HuggingFace, ShivamShrirao

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Import argparse first before initialising trainer for debug purposes first
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # Training Settings
    parser.add_argument("--resolution",                default=512, type=int, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    parser.add_argument("--train_batch_size",          default=4, type=int, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs",          default=1, type=int)
    parser.add_argument("--shuffle_per_epoch",         default=True, action="store_true", help="Will shffule the dataset per epoch")
    parser.add_argument("--use_bucketing",             default=False, action="store_true")
    parser.add_argument("--seed",                      default=42, type=int, help="A seed for reproducible training.")
    parser.add_argument("--learning_rate",             default=5e-6, type=float, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_scheduler",              default="constant", type=str, help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps",           default=500, type=float, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument('--token_limit',               default=75, type=int, help="Token limit, token lengths longer than the next multiple of 75 will be truncated.")
    parser.add_argument('--epoch_seed',                default=False, action="store_true")
    parser.add_argument("--min_snr_gamma",             default=None, type=float, help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper.")
    parser.add_argument('--with_pertubation_noise',    default=False, action="store_true")
    parser.add_argument("--perturbation_noise_weight", default=0.1, type=float, help="The weight of perturbation noise applied during training.")
    parser.add_argument("--zero_terminal_snr",         default=False, action="store_true", help="Enables Zero Terminal SNR, see https://arxiv.org/pdf/2305.08891.pdf - requires --force_v_pred for non SD2.1 models")
    parser.add_argument("--snr_debias",                default=False, action="store_true", help="Whether or not to debias timestep weighting during training, effectively replaces scale_v_pred_loss.")
    parser.add_argument("--force_v_pred",              default=False, action="store_true", help="Force enables V Prediction for models that don't officially support it - ie SD1.5")
    parser.add_argument("--scale_v_pred_loss",         default=False, action="store_true", help="By scaling the loss according to the time step, the weights of global noise prediction and local noise prediction become the same, and the improvement of details may be expected.")
    parser.add_argument("--unconditional_dropout",     default=-1, type=float,             help="A percentage of batches to drop out. 0 uses none, 1 uses all of them. Use -1 to disable. Batches will be duplicated with 'empty' captions.")

    # Model Settings
    parser.add_argument("--model_variant",                 default='base', type=str, help="Train Base/Inpaint/Depth2Img")
    parser.add_argument("--train_text_encoder",            default=False, action="store_true", help="Whether to train the text encoder")
    parser.add_argument("--stop_text_encoder_training",    default=999999999999999, type=int, help=("The epoch at which the text_encoder is no longer trained"))
    parser.add_argument('--clip_penultimate',              default=False, action="store_true", help='Use penultimate CLIP layer for text embedding')
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--pretrained_vae_name_or_path",   default=None, type=str, help="Path to pretrained vae or vae identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name",                default=None, type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument('--use_ema',                       default=False,action="store_true", help='Use EMA for finetuning')

    # Dataset Settings
    parser.add_argument("--concepts_list",                 default=None, type=str, help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.")
    parser.add_argument("--aspect_mode",                   default='dynamic', type=str, help="Sets how aspect buckets should have their dataset resized by.")
    parser.add_argument("--aspect_mode_action_preference", default='add', type=str, required=False, help="Override the preference from --aspect_mode.")
    parser.add_argument("--dataset_repeats",               default=1, type=int, help="repeat the dataset this many times")
    parser.add_argument('--use_text_files_as_captions',    default=False, action="store_true")
    parser.add_argument('--use_image_names_as_captions',   default=False, action="store_true")
    parser.add_argument("--auto_balance_concept_datasets", default=False, action="store_true", help="will balance the number of images in each concept dataset to match the minimum number of images in any concept dataset")
    
    # Optimisations
    parser.add_argument("--mixed_precision",             default="no", type=str, choices=["no", "fp16", "bf16","tf32"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--attention",                   default="xformers", type=str, choices=["xformers", "flash_attention"], help="Type of attention to use.")
    parser.add_argument('--disable_cudnn_benchmark',     default=True, action="store_true")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing",      default=False, action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--scale_lr",                    default=False, action="store_true", help="Scale the learning rate by the number of active GPUs and gradient accumulation steps.")
    parser.add_argument("--use_8bit_adam",               default=False, action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--use_lion",                    default=False, action="store_true", help="Whether or not to use Lion.")
    parser.add_argument("--adam_beta1",                  default=0.9, type=float, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2",                  default=0.999, type=float, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay",           default=1e-2, type=float, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon",                default=1e-08, type=float, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm",               default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--use_deepspeed_adam",          default=False, action="store_true", help="Use experimental DeepSpeed Adam 8.")
    parser.add_argument("--dataset_workers",             default=1, type=int, help="How many PyTorch threads should be used while preparing the dataset for latent caching.")
    parser.add_argument("--dataset_prefetch",            default=2, type=int, help="How many batches should PyTorch preload while preparing the dataset for latent caching.")

    # Misc Settings
    parser.add_argument("--save_every_n_epoch",      default=1, type=int, help="save on epoch finished")
    parser.add_argument("--save_every_quarter",      default=False, action="store_true", help="Saves the current epoch for every 25% of all steps completed.")
    parser.add_argument("--output_dir",              default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_train_steps",         default=None, type=int, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--regenerate_latent_cache", default=False, action="store_true")
    parser.add_argument('--use_latents_only',        default=False, action="store_true")
    parser.add_argument("--logging_dir",             default="logs", type=str, help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--log_interval",            default=10, type=int, help="Log every N steps.")
    parser.add_argument("--overwite_csv_logs",       default=False, action="store_true", help="Overwrites a CSV containing loss, LR, current step, current epoch and a timestamp when starting a new training session.")
    parser.add_argument("--local_rank",              default=-1, type=int, help="For distributed training: local_rank")
    parser.add_argument("--detect_full_drive",       default=True, action="store_true", help="Delete checkpoints when the drive is full.")
    parser.add_argument("--multi_gpu",               default=False, action="store_true", help="Enable this flag if you're using multi-GPU in HF Accelerate, otherwise, it's treated as a single GPU system.")
    parser.add_argument("--using_fsdp",              default=False, action="store_true", help="Enable this flag if you're using multi-GPU FSDP in HF Accelerate.")

    # Dreambooth Settings
    parser.add_argument("--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.")
    parser.add_argument("--prior_loss_weight",       default=1.0, type=float, help="The weight of prior preservation loss.")
    parser.add_argument("--num_class_images",        default=100, type=int, help="Minimal class images for prior preservation loss. If not have enough images, additional images will be sampled with class_prompt.")

    # Legacy Settings
    parser.add_argument("--center_crop",                 default=False, action="store_true", help="Whether to center crop images before resizing to resolution")
    parser.add_argument("--instance_data_dir",           default=None, type=str, help="A folder containing the training data of instance images.")
    parser.add_argument("--instance_prompt",             default=None, type=str, help="The prompt with identifier specifying the instance")
    parser.add_argument("--class_data_dir",              default=None, type=str, help="A folder containing the training data of class images.")
    parser.add_argument("--class_prompt",                default=None, type=str, help="The prompt to specify images in the same class as provided instance images.")
    parser.add_argument('--add_mask_prompt',             default=None, type=str, action="append", dest="mask_prompts")
    parser.add_argument("--sample_batch_size",           default=4, type=int, help="Batch size (per device) for sampling images.")
    parser.add_argument("--add_class_images_to_dataset", default=False, action="store_true", help="will generate and add class images to the dataset without using prior reservation in training")
    
    # Project Settings
    parser.add_argument("--project_name",                    default="model", type=str, help="The model name prefix. IE: yourmodel_rev_1")
    parser.add_argument("--project_append",                  default="", type=str, help="The model name suffix. IE: yourmodel_rev_1_e1_your_append")
    parser.add_argument("--half_completion_upload",          default=False, action="store_true", help="Whether to upload the current epoch at 50% completion to PixelDrain.")
    parser.add_argument("--webhook_user",                    default="StableTuner", type=str, help="The bot username for the Discord webhook.")
    parser.add_argument("--webhook_url",                     default="", type=str, help="The full URL for the Discord webhook. IE: https://discord.com/api/webhooks/....")
    parser.add_argument("--epoch_number",                    default="-1", type=int, help="Only used with constant_cosine when used within new launcher and when total epochs equal 1.")

    # Extra Settings
    parser.add_argument("--debug_flag",                    default=False, action="store_true", help="Used for development purposes. May exit, cause random effects or generate more debug logs than usual.")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

args = parse_args()

# Since we've passed any possible startup based debugging features - load the remaining dependancies:
import random
import hashlib
import itertools
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import shutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DiffusionPipeline, DPMSolverMultistepScheduler,EulerDiscreteScheduler
#from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from torchvision.transforms import functional
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, List, Generator, Tuple
from PIL import Image, ImageFile
from diffusers.utils.import_utils import is_xformers_available
from lion_pytorch import Lion
import trainer_util as tu
import subprocess
import safetensors
import requests
import time

from clip_segmentation import ClipSeg
import gc
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
logger = get_logger(__name__)

def add_ratio_to_bucket(x, y, bucket):
    bucket.append([x, y])
    bucket.append([y, x])

ASPECT_832 = [[832, 832], 
[896, 768], [768, 896], 
[960, 704], [704, 960], 
[1024, 640], [640, 1024], 
[1152, 576], [576, 1152], 
[1280, 512], [512, 1280], 
[1344, 512], [512, 1344], 
[1408, 448], [448, 1408], 
[1472, 448], [448, 1472], 
[1536, 384], [384, 1536], 
[1600, 384], [384, 1600]]

ASPECT_896 = [[896, 896],
[960, 832], [832, 960],
[1024, 768], [768, 1024],
[1088, 704], [704, 1088],
[1152, 704], [704, 1152],
[1216, 640], [640, 1216],
[1280, 640], [640, 1280],
[1344, 576], [576, 1344],
[1408, 576], [576, 1408],
[1472, 512], [512, 1472],
[1536, 512], [512, 1536], 
[1600, 448], [448, 1600], 
[1664, 448], [448, 1664]]

ASPECT_960 = [[896, 896], 
[960, 832], [832, 960], 
[1024, 768], [768, 1024], 
[1088, 704], [704, 1088], 
[1216, 640], [640, 1216], 
[1344, 576], [576, 1344], 
[1472, 512], [512, 1472], 
[1536, 512], [512, 1536], 
[1600, 448], [448, 1600], 
[1664, 448], [448, 1664]]

ASPECT_1024 = [[1024, 1024], 
[1088, 960], [960, 1088], 
[1152, 896], [896, 1152], 
[1216, 832], [832, 1216], 
[1344, 768], [768, 1344], 
[1472, 704], [704, 1472], 
[1600, 640], [640, 1600], 
[1728, 576], [576, 1728], 
[1792, 576], [576, 1792]]

ASPECT_768 = [[768,768],   # 589824 1:1
    [896,640],[640,896],   # 573440 1.4:1
    [832,704],[704,832],   # 585728 1.181:1
    [960,576],[576,960],   # 552960 1.6:1
    [1024,576],[576,1024], # 524288 1.778:1
    [1088,512],[512,1088], # 497664 2.125:1
    [1152,512],[512,1152], # 589824 2.25:1
    [1216,448],[448,1216], # 552960 2.714:1
    [1280,448],[448,1280], # 573440 2.857:1
    [1344,384],[384,1344], # 518400 3.5:1
    [1408,384],[384,1408], # 540672 3.667:1
    [1472,320],[320,1472], # 470400 4.6:1
    [1536,320],[320,1536], # 491520 4.8:1
]

ASPECT_704 = [[704,704]]                   # 495616 1:1
add_ratio_to_bucket(688, 720, ASPECT_704)  # 495360 1.05:1
add_ratio_to_bucket(672, 736, ASPECT_704)  # 494592 1.1:1
add_ratio_to_bucket(656, 752, ASPECT_704)  # 493312 1.15:1
add_ratio_to_bucket(640, 768, ASPECT_704)  # 491520 1.2:1
add_ratio_to_bucket(624, 784, ASPECT_704)  # 489216 1.25:1
add_ratio_to_bucket(616, 800, ASPECT_704)  # 492800 1.3:1
add_ratio_to_bucket(606, 818, ASPECT_704)  # 491264 1.33:1
add_ratio_to_bucket(592, 832, ASPECT_704)  # 492544 1.4:1
add_ratio_to_bucket(584, 848, ASPECT_704)  # 495232 1.45:1
add_ratio_to_bucket(568, 856, ASPECT_704)  # 486208 1.5:1
add_ratio_to_bucket(560, 872, ASPECT_704)  # 488320 1.55:1
add_ratio_to_bucket(552, 888, ASPECT_704)  # 490176 1.6:1
add_ratio_to_bucket(544, 904, ASPECT_704)  # 491776 1.66:1
add_ratio_to_bucket(536, 912, ASPECT_704)  # 488832 1.7:1
add_ratio_to_bucket(528, 936, ASPECT_704)  # 494208 1.778:1
add_ratio_to_bucket(512, 952, ASPECT_704)  # 487424 1.85:1
add_ratio_to_bucket(512, 968, ASPECT_704)  # 495616 1.9:1
add_ratio_to_bucket(496, 992, ASPECT_704)  # 492032 2:1
add_ratio_to_bucket(464, 1048, ASPECT_704) # 468272 2.25:1
add_ratio_to_bucket(440, 1104, ASPECT_704) # 485760 2.5:1
add_ratio_to_bucket(424, 1168, ASPECT_704) # 495232 2.75:1
add_ratio_to_bucket(400, 1208, ASPECT_704) # 483200 3:1
add_ratio_to_bucket(384, 1248, ASPECT_704) # 479232 3.25:1
add_ratio_to_bucket(368, 1288, ASPECT_704) # 473984 3.5:1
add_ratio_to_bucket(360, 1352, ASPECT_704) # 486720 3.75:1
add_ratio_to_bucket(352, 1408, ASPECT_704) # 495616 4:1
add_ratio_to_bucket(336, 1440, ASPECT_704) # 483840 4.25:1
add_ratio_to_bucket(328, 1488, ASPECT_704) # 488064 4.5:1
add_ratio_to_bucket(320, 1520, ASPECT_704) # 486400 4.75:1
add_ratio_to_bucket(312, 1520, ASPECT_704) # 486720 5:1
add_ratio_to_bucket(296, 1632, ASPECT_704) # 483072 5.5:1
add_ratio_to_bucket(280, 1680, ASPECT_704) # 470400 6:1
add_ratio_to_bucket(280, 1768, ASPECT_704) # 495040 6.25:1

ASPECT_640 = [[640,640],   # 409600 1:1
    [616,560],[560,616],   # 344960 1.1:1
    [680,544],[544,680],   # 369920 1.25:1
    [720,544],[544,720],   # 391680 1.33:1
    [768,512],[512,768],   # 393216 1.5:1
    [840,480],[480,840],   # 403200 1.75:1
    [896,448],[448,896],   # 401408 2:1
    [936,416],[416,936],   # 389376 2.25:1
    [1000,400],[400,1000], # 400000 2.5:1
    [1056,384],[384,1056], # 405504 2.75:1
    [1104,368],[368,1104], # 406272 3:1
    [1144,352],[352,1144], # 402688 3.25:1
    [1176,336],[336,1176], # 395136 3.5:1
    [1280,320],[320,1280], # 409600 4:1
    [1296,288],[288,1296], # 373248 4.5:1
    [1400,280],[280,1400], # 392000 5:1
    [1496,272],[272,1496], # 406912 5.5:1
    [1536,256],[256,1536], # 393216 6:1
    [1600,256],[256,1600], # 409600 6.25:1
]

ASPECT_1280 = [[1280,1280],  # 1638400 1:1
    [1320,1200],[1200,1320], # 1584000 1:1.1
    [1400,1120],[1120,1400], # 1568000 1:1.25
    [1392,1048],[1048,1392], # 1458816 1:1.33
    [1560,1040],[1040,1560], # 1622400 1:1.5
    [1680,960],[960,1680],   # 1612800 1:1.75
    [1808,904],[904,1808],   # 1634432 1:2
    [1872,832],[832,1872],   # 1557504 1:2.25
    [2000,800],[800,2000],   # 1600000 1:2.5
    [2112,768],[768,2112],   # 1622016 1:2.75
    [2208,736],[736,2208],   # 1625088 1:3
    [2288,704],[704,2288],   # 1610752 1:3.25
    [2352,672],[672,2352],   # 1580544 1:3.5
    [2560,640],[640,2560],   # 1638400 1:4
    [2664,592],[592,2664],   # 1577088 1:4.5
    [2840,568],[568,2840],   # 1613120 1:5
    [2992,544],[544,2992],   # 1627648 1:5.5
    [3120,520],[520,3120],   # 1622400 1:6
    [3200,512],[512,3200],   # 1638400 1:6.25
]

ASPECT_576 = [[576,576],     # 331776 1:1
    [640,512],[512,640],   # 327680 1.25:1
    [640,448],[448,640],   # 286720 1.4286:1
    [704,448],[448,704],   # 314928 1.5625:1
    [832,384],[384,832],   # 317440 2.1667:1
    [1024,320],[320,1024], # 327680 3.2:1
    [1280,256],[256,1280], # 327680 5:1
]

ASPECTS_512 = [[512,512],      # 262144 1:1
    [576,448],[448,576],   # 258048 1.29:1
    [640,384],[384,640],   # 245760 1.667:1
    [768,320],[320,768],   # 245760 2.4:1
    [832,256],[256,832],   # 212992 3.25:1
    [896,256],[256,896],   # 229376 3.5:1
    [960,256],[256,960],   # 245760 3.75:1
    [1024,256],[256,1024], # 245760 4:1
    ]

#failsafe aspects
ASPECTS = ASPECTS_512
def get_aspect_buckets(resolution,mode=''):
    if resolution < 512:
        raise ValueError("Resolution must be at least 512")
    try: 
        rounded_resolution = int(resolution / 64) * 64
        print(f"{bcolors.WARNING} Rounded resolution to: {rounded_resolution}{bcolors.ENDC}")   
        all_image_sizes = __get_all_aspects()
        if mode == 'MJ':
            #truncate to the first 3 resolutions
            all_image_sizes = [x[0:3] for x in all_image_sizes]
        aspects = next(filter(lambda sizes: sizes[0][0]==rounded_resolution, all_image_sizes), None)
        ASPECTS = aspects
        #print(aspects)
        return aspects
    except Exception as e:
        print(f"{bcolors.FAIL} *** Could not find selected resolution: {rounded_resolution}{bcolors.ENDC}")   

        raise e

def __get_all_aspects():
    return [ASPECTS_512, ASPECT_576, ASPECT_640, ASPECT_704, ASPECT_768,ASPECT_832,ASPECT_896,ASPECT_960,ASPECT_1024,ASPECT_1280]

class AutoBucketing(Dataset):
    def __init__(self,
                    concepts_list,
                    tokenizer=None,
                    flip_p=0.0,
                    repeats=1,
                    debug_level=0,
                    batch_size=1,
                    set='val',
                    resolution=512,
                    center_crop=False,
                    use_image_names_as_captions=True,
                    add_class_images_to_dataset=None,
                    balance_datasets=False,
                    crop_jitter=20,
                    with_prior_loss=False,
                    use_text_files_as_captions=False,
                    aspect_mode='dynamic',
                    action_preference='dynamic',
                    seed=555,
                    model_variant='base',
                    extra_module=None,
                    mask_prompts=None,
                    ):
        
        self.debug_level = debug_level
        self.resolution = resolution
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.concepts_list = concepts_list
        self.use_image_names_as_captions = use_image_names_as_captions
        self.num_train_images = 0
        self.num_reg_images = 0
        self.image_train_items = []
        self.image_reg_items = []
        self.add_class_images_to_dataset = add_class_images_to_dataset
        self.balance_datasets = balance_datasets
        self.crop_jitter = crop_jitter
        self.with_prior_loss = with_prior_loss
        self.use_text_files_as_captions = use_text_files_as_captions
        self.aspect_mode = aspect_mode
        self.action_preference = action_preference
        self.model_variant = model_variant
        self.extra_module = extra_module
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.depth_image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.seed = seed
        #shared_dataloader = None
        print(f"{bcolors.WARNING}Creating Auto Bucketing Dataloader{bcolors.ENDC}")

        shared_dataloader = DataLoaderMultiAspect(concepts_list,
         debug_level=debug_level,
         resolution=self.resolution,
         seed=self.seed,
         batch_size=self.batch_size, 
         flip_p=flip_p,
         use_image_names_as_captions=self.use_image_names_as_captions,
         add_class_images_to_dataset=self.add_class_images_to_dataset,
         balance_datasets=self.balance_datasets,
         with_prior_loss=self.with_prior_loss,
         use_text_files_as_captions=self.use_text_files_as_captions,
         aspect_mode=self.aspect_mode,
         action_preference=self.action_preference,
         model_variant=self.model_variant,
         extra_module=self.extra_module,
         mask_prompts=mask_prompts,
        )

        #print(self.image_train_items)
        if self.with_prior_loss and self.add_class_images_to_dataset == False:
            self.image_train_items, self.class_train_items = shared_dataloader.get_all_images()
            self.num_train_images = self.num_train_images + len(self.image_train_items)
            self.num_reg_images = self.num_reg_images + len(self.class_train_items)
            self._length = max(max(math.trunc(self.num_train_images * repeats), batch_size),math.trunc(self.num_reg_images * repeats), batch_size) - self.num_train_images % self.batch_size
            self.num_train_images = self.num_train_images + self.num_reg_images
            
        else:
            self.image_train_items = shared_dataloader.get_all_images()
            self.num_train_images = self.num_train_images + len(self.image_train_items)
            self._length = max(math.trunc(self.num_train_images * repeats), batch_size) - self.num_train_images % self.batch_size

        print()
        print(f"{bcolors.WARNING} ** Validation Set: {set}, steps: {self._length / batch_size:.0f}, repeats: {repeats} {bcolors.ENDC}")
        print()

    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_train_images
        #print(idx)
        image_train_item = self.image_train_items[idx]
        
        example = self.__get_image_for_trainer(image_train_item,debug_level=self.debug_level)
        if self.with_prior_loss and self.add_class_images_to_dataset == False:
            idx = i % self.num_reg_images
            class_train_item = self.class_train_items[idx]
            example_class = self.__get_image_for_trainer(class_train_item,debug_level=self.debug_level,class_img=True)
            example= {**example, **example_class}
            
        #print the tensor shape
        #print(example['instance_images'].shape)
        #print(example.keys())
        return example

    def normalize8(self,I):
            mn = I.min()
            mx = I.max()

            mx -= mn
            # Avoid division by zero
            I = (I - mn)/max(0.003922, mx) * 255
            return I.astype(np.uint8)

    def __get_image_for_trainer(self,image_train_item,debug_level=0,class_img=False):
        example = {}
        save = debug_level > 2
        
        if class_img==False:
            image_train_tmp = image_train_item.hydrate(crop=False, save=0, crop_jitter=self.crop_jitter)
            image_train_tmp_image = Image.fromarray(self.normalize8(image_train_tmp.image)).convert("RGB")
            
            example["instance_images"] = self.image_transforms(image_train_tmp_image)
            if self.model_variant == 'inpainting':
                image_train_tmp_mask = Image.fromarray(self.normalize8(image_train_tmp.extra)).convert("L")
                example["mask"] = self.mask_transforms(image_train_tmp_mask)
            if self.model_variant == 'depth2img':
                image_train_tmp_depth = Image.fromarray(self.normalize8(image_train_tmp.extra)).convert("L")
                example["instance_depth_images"] = self.depth_image_transforms(image_train_tmp_depth)
            #print(image_train_tmp.caption)
            example["instance_prompt_ids"] = self.tokenizer(
                image_train_tmp.caption,
                padding="do_not_pad",
                verbose=False
            ).input_ids
            image_train_item.self_destruct()
            return example

        if class_img==True:
            image_train_tmp = image_train_item.hydrate(crop=False, save=4, crop_jitter=self.crop_jitter)
            image_train_tmp_image = Image.fromarray(self.normalize8(image_train_tmp.image)).convert("RGB")
            if self.model_variant == 'depth2img':
                image_train_tmp_depth = Image.fromarray(self.normalize8(image_train_tmp.extra)).convert("L")
                example["class_depth_images"] = self.depth_image_transforms(image_train_tmp_depth)
            example["class_images"] = self.image_transforms(image_train_tmp_image)
            example["class_prompt_ids"] = self.tokenizer(
                image_train_tmp.caption,
                padding="do_not_pad",
                verbose=False
            ).input_ids
            image_train_item.self_destruct()
            return example

_RANDOM_TRIM = 0.04
class ImageTrainItem(): 
    """
    image: Image
    mask: Image
    identifier: caption,
    target_aspect: (width, height), 
    pathname: path to image file
    flip_p: probability of flipping image (0.0 to 1.0)
    """    
    def __init__(self, image: Image, extra: Image, caption: str, target_wh: list, pathname: str, flip_p=0.0, model_variant='base'):
        self.caption = caption
        self.target_wh = target_wh
        self.pathname = pathname
        self.mask_pathname = os.path.splitext(pathname)[0] + "-masklabel.png"
        self.depth_pathname = os.path.splitext(pathname)[0] + "-depth.png"
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropped_img = None
        self.model_variant = model_variant
        self.is_dupe = []
        self.variant_warning = False

        if image is None:
            self.image = []
        else:
            self.image = image

        if extra is None:
            self.extra = []
        else:
            self.extra = extra

    def self_destruct(self):
        self.image = []
        self.extra = []
        self.cropped_img = None
        self.is_dupe.append(1)

    def load_image(self, pathname, crop, crop_jitter):
        if len(self.is_dupe) > 0:
            chance = float(len(self.is_dupe)) / 10.0
            self.flip = transforms.RandomHorizontalFlip(p=self.flip_p + chance if chance < 1.0 else 1.0)
            self.crop_jitter = crop_jitter + (len(self.is_dupe) * 10) if crop_jitter < 50 else 50
        image = Image.open(pathname).convert('RGB')

        width, height = image.size
        if crop:
            cropped_img = self.__autocrop(image)
            image = cropped_img.resize((512, 512), resample=Image.Resampling.LANCZOS)
        else:
            width, height = image.size
            jitter_amount = random.randint(0, crop_jitter)

            if self.target_wh[0] == self.target_wh[1]:
                if width > height:
                    left = random.randint(0, width - height)
                    image = image.crop((left, 0, height + left, height))
                    width = height
                elif height > width:
                    top = random.randint(0, height - width)
                    image = image.crop((0, top, width, width + top))
                    height = width
                elif width > self.target_wh[0]:
                    slice = min(int(self.target_wh[0] * _RANDOM_TRIM), width - self.target_wh[0])
                    slicew_ratio = random.random()
                    left = int(slice * slicew_ratio)
                    right = width - int(slice * (1 - slicew_ratio))
                    sliceh_ratio = random.random()
                    top = int(slice * sliceh_ratio)
                    bottom = height - int(slice * (1 - sliceh_ratio))

                    image = image.crop((left, top, right, bottom))
            else:
                image_aspect = width / height
                target_aspect = self.target_wh[0] / self.target_wh[1]
                if image_aspect > target_aspect:
                    new_width = int(height * target_aspect)
                    jitter_amount = max(min(jitter_amount, int(abs(width - new_width) / 2)), 0)
                    left = jitter_amount
                    right = left + new_width
                    image = image.crop((left, 0, right, height))
                else:
                    new_height = int(width / target_aspect)
                    jitter_amount = max(min(jitter_amount, int(abs(height - new_height) / 2)), 0)
                    top = jitter_amount
                    bottom = top + new_height
                    image = image.crop((0, top, width, bottom))
                    # LAZCOS resample
            image = image.resize(self.target_wh, resample=Image.Resampling.LANCZOS)
            # print the pixel count of the image
            # print path to image file
            # print(self.pathname)
            # print(self.image.size[0] * self.image.size[1])
            image = self.flip(image)
        return image

    def hydrate(self, crop=False, save=False, crop_jitter=20):
        """
        crop: hard center crop to 512x512
        save: save the cropped image to disk, for manual inspection of resize/crop
        crop_jitter: randomly shift cropp by N pixels when using multiple aspect ratios to improve training quality
        """

        if not hasattr(self, 'image') or len(self.image) == 0:
            self.image = self.load_image(self.pathname, crop, crop_jitter)
            if self.model_variant == "inpainting":
                if os.path.exists(self.mask_pathname):
                    self.extra = self.load_image(self.mask_pathname, crop, crop_jitter)
                else:
                    if self.variant_warning == False:
                        print(f"{bcolors.FAIL} ** Warning: No mask found for an image, using an empty mask but make sure you're training the right model variant.{bcolors.ENDC}")
                        self.variant_warning = True
                    self.extra = Image.new('RGB', self.image.size, color="white").convert("L")
            if self.model_variant == "depth2img":
                if os.path.exists(self.depth_pathname):
                    self.extra = self.load_image(self.depth_pathname, crop, crop_jitter)
                else:
                    if self.variant_warning == False:
                        print(f"{bcolors.FAIL} ** Warning: No depth found for an image, using an empty depth but make sure you're training the right model variant.{bcolors.ENDC}")
                        self.variant_warning = True
                    self.extra = Image.new('RGB', self.image.size, color="white").convert("L")
        if type(self.image) is not np.ndarray:
            if save: 
                base_name = os.path.basename(self.pathname)
                if not os.path.exists("test/output"):
                    os.makedirs("test/output")
                self.image.save(f"test/output/{base_name}")
            
            self.image = np.array(self.image).astype(np.uint8)

            self.image = (self.image / 127.5 - 1.0).astype(np.float32)
        if self.model_variant != "base":
            if type(self.extra) is not np.ndarray:
                self.extra = np.array(self.extra).astype(np.uint8)

                self.extra = (self.extra / 255.0).astype(np.float32)

        #print(self.image.shape)

        return self

class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing
    data_root: root folder of training data
    batch_size: number of images per batch
    flip_p: probability of flipping image horizontally (i.e. 0-0.5)
    """
    def __init__(
            self,
            concept_list,
            seed=555,
            debug_level=0,
            resolution=512,
            batch_size=1,
            flip_p=0.0,
            use_image_names_as_captions=True,
            add_class_images_to_dataset=False,
            balance_datasets=False,
            with_prior_loss=False,
            use_text_files_as_captions=False,
            aspect_mode='dynamic',
            action_preference='add',
            model_variant='base',
            extra_module=None,
            mask_prompts=None,
    ):
        self.resolution = resolution
        self.debug_level = debug_level
        self.flip_p = flip_p
        self.use_image_names_as_captions = use_image_names_as_captions
        self.balance_datasets = balance_datasets
        self.with_prior_loss = with_prior_loss
        self.add_class_images_to_dataset = add_class_images_to_dataset
        self.use_text_files_as_captions = use_text_files_as_captions
        self.aspect_mode = aspect_mode
        self.action_preference = action_preference
        self.seed = seed
        self.model_variant = model_variant
        self.extra_module = extra_module
        self.prepared_train_data = []
        
        self.aspects = get_aspect_buckets(resolution)
        #print(f"* DLMA resolution {resolution}, buckets: {self.aspects}")
        #process sub directories flag
            
        print(f"{bcolors.WARNING} Preloading images...{bcolors.ENDC}")   

        for concept in concept_list:
            if 'use_sub_dirs' in concept:
                if concept['use_sub_dirs'] == True:
                    use_sub_dirs = True
                else:
                    use_sub_dirs = False
            else:
                use_sub_dirs = False
            self.image_paths = []
            #self.class_image_paths = []
            min_concept_num_images = None
            data_root = concept['instance_data_dir']
            concept_prompt = concept['instance_prompt']
            data_root_class = ""
            concept_class_prompt = ""
            if 'flip_p' in concept.keys():
                flip_p = concept['flip_p']
                if flip_p == '':
                    flip_p = 0.0
                else:
                    flip_p = float(flip_p)
            self.flip_p = flip_p

            self.concept_prompt = concept_prompt
            self.__recurse_data_root(self=self, recurse_root=data_root,use_sub_dirs=use_sub_dirs)
            random.Random(self.seed).shuffle(self.image_paths)
            
        self.image_caption_pairs = self.__bucketize_images(self.prepared_train_data, batch_size=batch_size, debug_level=debug_level,aspect_mode=self.aspect_mode,action_preference=self.action_preference)
        #print the length of image_caption_pairs
        print(f"{bcolors.WARNING} Number of image-caption pairs: {len(self.image_caption_pairs)}{bcolors.ENDC}") 
        if len(self.image_caption_pairs) == 0:
            raise Exception("All the buckets are empty. Please check your data or reduce the batch size.")
    def get_all_images(self):
        if self.with_prior_loss == False:
            return self.image_caption_pairs
        else:
            return self.image_caption_pairs, self.class_image_caption_pairs

    @staticmethod
    def __bucketize_images(prepared_train_data: list, batch_size=1, debug_level=0,aspect_mode='dynamic',action_preference='add'):
        """
        Put images into buckets based on aspect ratio with batch_size*n images per bucket, discards remainder
        """

        # TODO: this is not terribly efficient but at least linear time
        buckets = {}
        for image_caption_pair in prepared_train_data:
            target_wh = image_caption_pair.target_wh

            if (target_wh[0],target_wh[1]) not in buckets:
                buckets[(target_wh[0],target_wh[1])] = []
            buckets[(target_wh[0],target_wh[1])].append(image_caption_pair)
        print(f" ** Number of buckets: {len(buckets)}")
        for bucket in buckets:
            bucket_len = len(buckets[bucket])
            #real_len = len(buckets[bucket])+1
            #print(real_len)
            truncate_amount = bucket_len % batch_size
            add_amount = batch_size - bucket_len % batch_size
            action = None
            bratio = ""
            bmode = ""
            if bucket[0] <= bucket[1]:
                bratio = bucket[1] / bucket[0]
                if bratio == 1:
                    bmode = f"(1:1)"
                else:
                    bmode = f"(1:{bratio:.2f})"
            else:
                bratio = bucket[0] / bucket[1]
                bmode = f"({bratio:.2f}:1)"
            #print(f" ** Bucket {bucket} has {bucket_len} images")
            if aspect_mode == 'dynamic':
                if batch_size == bucket_len:
                    action = None
                elif add_amount < truncate_amount and add_amount != 0 and add_amount != batch_size or truncate_amount == 0:
                    action = 'add'
                    #print(f'should add {add_amount}')
                elif truncate_amount < add_amount and truncate_amount != 0 and truncate_amount != batch_size and batch_size < bucket_len:
                    #print(f'should truncate {truncate_amount}')
                    action = 'truncate'
                    #truncate the bucket
                elif truncate_amount == add_amount:
                    if action_preference == 'add':
                        action = 'add'
                    elif action_preference == 'truncate':
                        action = 'truncate'
                elif batch_size > bucket_len:
                    action = 'add'

            elif aspect_mode == 'add':
                action = 'add'
            elif aspect_mode == 'truncate':
                action = 'truncate'
            if action == None:
                action = None
                #print('no need to add or truncate')
            if action == None:
                #print('test')
                current_bucket_size = bucket_len
                print(f"  ** Bucket {bucket} {bmode} found {bucket_len}, nice!")
            elif action == 'add':
                #copy the bucket
                shuffleBucket = random.sample(buckets[bucket], bucket_len)
                #add the images to the bucket
                current_bucket_size = bucket_len
                truncate_count = (bucket_len) % batch_size
                #how many images to add to the bucket to fill the batch
                addAmount = batch_size - truncate_count
                if addAmount != batch_size:
                    added=0
                    while added != addAmount:
                        randomIndex = random.randint(0,len(shuffleBucket)-1)
                        #print(str(randomIndex))
                        buckets[bucket].append(shuffleBucket[randomIndex])
                        added+=1
                    print(f"  ** Bucket {bucket} {bmode} found {bucket_len}  images, will {bcolors.OKCYAN}duplicate {added} images{bcolors.ENDC} due to batch size {bcolors.WARNING}{batch_size}{bcolors.ENDC}")
                else:
                    print(f"  ** Bucket {bucket} {bmode} found {bucket_len}, {bcolors.OKGREEN}nice!{bcolors.ENDC}")
            elif action == 'truncate':
                truncate_count = (bucket_len) % batch_size
                current_bucket_size = bucket_len
                buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]
                print(f"  ** Bucket {bucket} found {bucket_len} ({bmode}) images, will {bcolors.FAIL}drop {truncate_count} images{bcolors.ENDC} due to batch size {bcolors.WARNING}{batch_size}{bcolors.ENDC}")
            

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])

        return image_caption_pairs

    @staticmethod
    def __recurse_data_root(self, recurse_root,use_sub_dirs=True,class_images=False,concept=None):
        progress_bar = tqdm(os.listdir(recurse_root), desc=f"{bcolors.WARNING} ** Processing {recurse_root}{bcolors.ENDC}")
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)
            if os.path.isfile(current):
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    #try to open the file to make sure it's a valid image
                    try:
                        img = Image.open(current)

                        identifier = self.concept_prompt
                        if self.use_image_names_as_captions:
                            caption_from_filename = os.path.splitext(os.path.basename(current))[0].split("_")[0]
                            identifier = caption_from_filename
                        if self.use_text_files_as_captions:
                            txt_file_path = os.path.splitext(current)[0] + ".txt"
                            if os.path.exists(txt_file_path):
                                try:
                                    with open(txt_file_path, 'r',encoding='utf-8',errors='ignore') as txt:
                                        identifier = txt.readline().strip()
                                        txt.close()
                                        if len(identifier) < 1:
                                            raise ValueError(f" *** Could not find valid text in: {txt_file_path}")
                                except Exception as e:
                                    print(f"{bcolors.FAIL} *** Error reading {txt_file_path} to get caption, falling back to filename{bcolors.ENDC}") 
                                    print(e)
                                    identifier = caption_from_filename
                        
                        width, height = img.size
                        image_aspect = width / height
                        target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

                        image_train_item = ImageTrainItem(image=None, extra=None, caption=identifier, target_wh=target_wh, pathname=current, flip_p=self.flip_p,model_variant=self.model_variant)
                        self.prepared_train_data.append(image_train_item)
                    except:
                        print(f" ** Skipping {current} because it failed to open, please check the file")
                        progress_bar.update(1)
                        continue
                    del img
                    if class_images == False:
                        self.image_paths.append(current)
                    else:
                        self.class_images_path.append(current)
            progress_bar.update(1)
        if use_sub_dirs:
            sub_dirs = []

            for d in os.listdir(recurse_root):
                current = os.path.join(recurse_root, d)
                if os.path.isdir(current):
                    sub_dirs.append(current)

            for dir in sub_dirs:
                self.__recurse_data_root(self=self, recurse_root=dir)

class NormalDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        with_prior_preservation=True,
        size=512,
        center_crop=False,
        num_class_images=None,
        use_image_names_as_captions=False,
        repeats=1,
        use_text_files_as_captions=False,
        seed=555,
        model_variant='base',
        extra_module=None,
        mask_prompts=None,
    ):
        self.use_image_names_as_captions = use_image_names_as_captions
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.use_text_files_as_captions = use_text_files_as_captions
        self.image_paths = []
        self.class_images_path = []
        self.seed = seed
        self.model_variant = model_variant
        self.variant_warning = False
        self.vae_scale_factor = None
        for concept in concepts_list:
            if 'use_sub_dirs' in concept:
                if concept['use_sub_dirs'] == True:
                    use_sub_dirs = True
                else:
                    use_sub_dirs = False
            else:
                use_sub_dirs = False

            for i in range(repeats):
                self.__recurse_data_root(self, concept,use_sub_dirs=use_sub_dirs)

            if with_prior_preservation:
                for i in range(repeats):
                    self.__recurse_data_root(self, concept,use_sub_dirs=False,class_images=True)
        if self.model_variant == "inpainting" and mask_prompts is not None:
            print(f"{bcolors.WARNING} Checking and generating missing masks{bcolors.ENDC}")
            clip_seg = ClipSeg()
            clip_seg.mask_images(self.image_paths, mask_prompts)
            del clip_seg

        random.Random(seed).shuffle(self.image_paths)
        self.num_instance_images = len(self.image_paths)
        self._length = self.num_instance_images
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        if self.model_variant == 'depth2img':
            print(f"{bcolors.WARNING} ** Loading Depth2Img Pipeline To Process Dataset{bcolors.ENDC}")
            self.vae_scale_factor = extra_module.depth_images(self.image_paths)
            if self.with_prior_preservation:
                print(f"{bcolors.WARNING} ** Loading Depth2Img Class Processing{bcolors.ENDC}")
                extra_module.depth_images(self.class_images_path)
        print(f"{bcolors.WARNING} ** Dataset length: {self._length}, {int(self.num_instance_images / repeats)} images using {repeats} repeats{bcolors.ENDC}")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
            
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ])

        self.depth_image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def __recurse_data_root(self, recurse_root,use_sub_dirs=True,class_images=False):
        #if recurse root is a dict
        if isinstance(recurse_root, dict):
            if class_images == True:
                #print(f"{bcolors.WARNING} ** Processing class images: {recurse_root['class_data_dir']}{bcolors.ENDC}")
                concept_token = recurse_root['class_prompt']
                data = recurse_root['class_data_dir']
            else:
                #print(f"{bcolors.WARNING} ** Processing instance images: {recurse_root['instance_data_dir']}{bcolors.ENDC}")
                concept_token = recurse_root['instance_prompt']
                data = recurse_root['instance_data_dir']


        else:
            concept_token = None
        #progress bar
        progress_bar = tqdm(os.listdir(data), desc=f"{bcolors.WARNING} ** Processing {data}{bcolors.ENDC}")
        for f in os.listdir(data):
            current = os.path.join(data, f)
            if os.path.isfile(current):
                if '-depth' in f or '-masklabel' in f:
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] and '-masklabel.png' not in f:
                    try:
                        img = Image.open(current)
                    except:
                        print(f" ** Skipping {current} because it failed to open, please check the file")
                        progress_bar.update(1)
                        continue
                    del img
                    if class_images == False:
                        self.image_paths.append([current,concept_token])
                    else:
                        self.class_images_path.append([current,concept_token])
            progress_bar.update(1)
        if use_sub_dirs:
            sub_dirs = []

            for d in os.listdir(data):
                current = os.path.join(data, d)
                if os.path.isdir(current):
                    sub_dirs.append(current)

            for dir in sub_dirs:
                if class_images == False:
                    self.__recurse_data_root(self=self, recurse_root={'instance_data_dir' : dir, 'instance_prompt' : concept_token})
                else:
                    self.__recurse_data_root(self=self, recurse_root={'class_data_dir' : dir, 'class_prompt' : concept_token})
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.image_paths[index % self.num_instance_images]
        og_prompt = instance_prompt
        instance_image = Image.open(instance_path)
        if self.model_variant == "inpainting":

            mask_pathname = os.path.splitext(instance_path)[0] + "-masklabel.png"
            if os.path.exists(mask_pathname):
                mask = Image.open(mask_pathname).convert("L")
            else:
                if self.variant_warning == False:
                    print(f"{bcolors.FAIL} ** Warning: No mask found for an image, using an empty mask but make sure you're training the right model variant.{bcolors.ENDC}")
                    self.variant_warning = True
                size = instance_image.size
                mask = Image.new('RGB', size, color="white").convert("L")
            example["mask"] = self.mask_transforms(mask)
        if self.model_variant == "depth2img":
            depth_pathname = os.path.splitext(instance_path)[0] + "-depth.png"
            if os.path.exists(depth_pathname):
                depth_image = Image.open(depth_pathname).convert("L")
            else:
                if self.variant_warning == False:
                    print(f"{bcolors.FAIL} ** Warning: No depth image found for an image, using an empty depth image but make sure you're training the right model variant.{bcolors.ENDC}")
                    self.variant_warning = True
                size = instance_image.size
                depth_image = Image.new('RGB', size, color="white").convert("L")
            example["instance_depth_images"] = self.depth_image_transforms(depth_image)

        if self.use_image_names_as_captions == True:
            instance_prompt = str(instance_path).split(os.sep)[-1].split('.')[0].split('_')[0]
        #else if there's a txt file with the same name as the image, read the caption from there
        if self.use_text_files_as_captions == True:
            #if there's a file with the same name as the image, but with a .txt extension, read the caption from there
            #get the last . in the file name
            last_dot = str(instance_path).rfind('.')
            #get the path up to the last dot
            txt_path = str(instance_path)[:last_dot] + '.txt'

            #if txt_path exists, read the caption from there
            if os.path.exists(txt_path):
                with open(txt_path, encoding='utf-8') as f:
                    instance_prompt = f.readline().rstrip()
                    f.close()
                
            
        #print('identifier: ' + instance_prompt)
        instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            verbose=False
        ).input_ids
        if self.with_prior_preservation:
            class_path, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            if self.model_variant == "inpainting":
                mask_pathname = os.path.splitext(class_path)[0] + "-masklabel.png"
                if os.path.exists(mask_pathname):
                    mask = Image.open(mask_pathname).convert("L")
                else:
                    if self.variant_warning == False:
                        print(f"{bcolors.FAIL} ** Warning: No mask found for an image, using an empty mask but make sure you're training the right model variant.{bcolors.ENDC}")
                        self.variant_warning = True
                    size = instance_image.size
                    mask = Image.new('RGB', size, color="white").convert("L")
                example["class_mask"] = self.mask_transforms(mask)
            if self.model_variant == "depth2img":
                depth_pathname = os.path.splitext(class_path)[0] + "-depth.png"
                if os.path.exists(depth_pathname):
                    depth_image = Image.open(depth_pathname)
                else:
                    if self.variant_warning == False:
                        print(f"{bcolors.FAIL} ** Warning: No depth image found for an image, using an empty depth image but make sure you're training the right model variant.{bcolors.ENDC}")
                        self.variant_warning = True
                    size = instance_image.size
                    depth_image = Image.new('RGB', size, color="white").convert("L")
                example["class_depth_images"] = self.depth_image_transforms(depth_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="do_not_pad",
                verbose=False
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class CachedLatentsDataset(Dataset):
    #stores paths and loads latents on the fly
    def __init__(self, cache_paths=[],batch_size=None,tokenizer=None,text_encoder=None,dtype=None,model_variant='base',shuffle_per_epoch=False,args=None,accelerator=None):
        self.cache_paths = cache_paths
        self.tokenizer = tokenizer
        self.args = args
        self.text_encoder = text_encoder
        self.accelerator = accelerator
        #get text encoder device
        text_encoder_device = next(self.text_encoder.parameters()).device
        self.empty_batch = [self.tokenizer('',padding="do_not_pad",truncation=True,max_length=self.tokenizer.model_max_length,).input_ids for i in range(batch_size)]
        #handle text encoder for empty tokens
        self.empty_tokens = tokenizer.pad({"input_ids": self.empty_batch},padding="max_length",max_length=tokenizer.model_max_length,return_tensors="pt",).to(accelerator.device).input_ids
        self.empty_tokens.to(accelerator.device, dtype=dtype)

        self.model_variant = model_variant
        self.shuffle_per_epoch = shuffle_per_epoch
    def __len__(self):
        return len(self.cache_paths)
    def __getitem__(self, index):
        if index == 0:
            if self.shuffle_per_epoch == True:
                random.shuffle(self.cache_paths)

        self.cache = torch.load(self.cache_paths[index][0])
        self.latents = self.cache.latents_cache[0]
        self.tokens = self.cache.tokens_cache[0]
        self.conditioning_latent_cache = None
        self.extra_cache = None
        if self.cache_paths[index][1]:
            self.text_encoder = self.empty_tokens.to(self.accelerator.device)
        else:
            self.text_encoder = self.cache.text_encoder_cache[0].to(self.accelerator.device)

        if self.model_variant != 'base':
            self.conditioning_latent_cache = self.cache.conditioning_latent_cache[0]
            self.extra_cache = self.cache.extra_cache[0]
        del self.cache
        return self.latents, self.text_encoder, self.conditioning_latent_cache, self.extra_cache, self.tokens

    def get_cache_list(self):
        return self.cache_paths

    def add_pt_cache(self, cache_path, dropout=False):
        if dropout:
            self.cache_paths.append((cache_path, True))
        else:
            self.cache_paths.append((cache_path, False))


class LatentsDataset(Dataset):
    def __init__(self, latents_cache=None, text_encoder_cache=None, conditioning_latent_cache=None, extra_cache=None, tokens_cache=None):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache
        self.conditioning_latent_cache = conditioning_latent_cache
        self.extra_cache = extra_cache
        self.tokens_cache = tokens_cache
    def add_latent(self, latent, text_encoder, cached_conditioning_latent, cached_extra, tokens_cache):
        self.latents_cache.append(latent)
        self.text_encoder_cache.append(text_encoder)
        self.conditioning_latent_cache.append(cached_conditioning_latent)
        self.extra_cache.append(cached_extra)
        self.tokens_cache.append(tokens_cache)
    def __len__(self):
        return len(self.latents_cache)
    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index], self.conditioning_latent_cache[index], self.extra_cache[index], self.tokens_cache[index]
class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def main():
    print(f"{bcolors.OKBLUE}Booting Up StableTuner{bcolors.ENDC}") 
    print(f"{bcolors.OKBLUE}Please wait a moment as we load up some stuff...{bcolors.ENDC}") 
    #torch.cuda.set_per_process_memory_fraction(0.5)
    if args.disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    logging_dir = Path(args.output_dir, "logs", args.logging_dir)
    main_sample_dir = os.path.join(args.output_dir, "samples")
    if os.path.exists(main_sample_dir):
            shutil.rmtree(main_sample_dir)
            os.makedirs(main_sample_dir)
    #create logging directory
    if not logging_dir.exists():
        logging_dir.mkdir(parents=True)
    #create output directory
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != 'tf32' else 'no',
        log_with="tensorboard",
        project_dir=logging_dir,
    )

    # Torch memory debugging
    torch_mem_output = os.path.join(args.output_dir, "logs/", f"torch_main_process_memory_id_{accelerator.process_index}.pickle")
    if args.debug_flag:
        print(f"{bcolors.WARNING}Recording memory for GPU process id {accelerator.process_index}.{bcolors.ENDC}")
        torch.cuda.memory._record_memory_history(max_entries=100000)
    else:
        torch.cuda.memory._record_memory_history(enabled=None)


    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    if args.with_prior_preservation or args.add_class_images_to_dataset:
        pipeline = None
        for concept in args.concepts_list:
            class_images_dir = Path(concept["class_data_dir"])
            class_images_dir.mkdir(parents=True, exist_ok=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if pipeline is None:

                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae" ,safe_serialization=True),
                        torch_dtype=torch_dtype,
                        requires_safety_checker=False
                    )
                    pipeline.set_progress_bar_config(disable=True)
                    pipeline.to(accelerator.device)
                
                #if args.use_bucketing == False:
                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(concept["class_prompt"], num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
                sample_dataloader = accelerator.prepare(sample_dataloader)
                #else:
                    #create class images that match up to the concept target buckets
                #    instance_images_dir = Path(concept["instance_data_dir"])
                #    cur_instance_images = len(list(instance_images_dir.iterdir()))
                    #target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))
                #    num_new_images = cur_instance_images - cur_class_images
                
                

                with torch.autocast("cuda"):
                    for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        with torch.autocast("cuda"):
                            images = pipeline(example["prompt"],height=args.resolution,width=args.resolution).images
                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name )
    elif args.pretrained_model_name_or_path:
        #print(os.getcwd())
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer" )

    # Load models and create wrapper for stable diffusion
    #text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
    text_encoder_cls = tu.import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet" )
    if is_xformers_available() and args.attention=='xformers':
        try:
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    elif args.attention=='flash_attention':
        tu.replace_unet_cross_attn_to_flash_attention()
    if args.use_ema == True:
        ema_unet = tu.EMAModel(unet.parameters())
        # This apparently saves a lot of memory
        #for param in ema_unet.parameters():
            #param.requires_grad = False
    if args.model_variant == "depth2img":
        d2i = tu.Depth2Img(unet,text_encoder,args.mixed_precision,args.pretrained_model_name_or_path,accelerator)
    vae.requires_grad_(False)
    vae.enable_slicing()
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr and args.multi_gpu:
        args.learning_rate *= accelerator.num_processes
        print(f"{bcolors.WARNING}LR rescaled to {args.learning_rate} for multiple GPUs!{bcolors.ENDC}")

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam and args.use_deepspeed_adam==False:
        try:
            import bitsandbytes as bnb
            print("Using 8bitAdam")
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    elif args.use_8bit_adam and args.use_deepspeed_adam==True:
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError:
            raise ImportError(
                "To use 8-bit DeepSpeed Adam, try updating your cuda and deepspeed integrations."
            )
        
        optimizer_class = DeepSpeedCPUAdam
    elif args.use_lion:
        print("Using LION optimizer")
        optimizer_class = Lion
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    if args.use_lion == False:
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay
        )
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    #noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.zero_terminal_snr:
        print(f"{bcolors.WARNING}Enforcing Zero Terminal SNR.{bcolors.ENDC}")
        noise_scheduler.betas = tu.enforce_zero_terminal_snr(noise_scheduler.betas)
        tu.prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

    if args.with_pertubation_noise:
        print(f"{bcolors.WARNING}Perturbation Noise set to {args.perturbation_noise_weight}.{bcolors.ENDC}")

    if args.use_latents_only:
        print(f"{bcolors.WARNING}Notice: Running from latent cache only!.{bcolors.ENDC}")
    elif not args.use_latents_only or args.regenerate_latent_cache:
        if args.use_bucketing:
            train_dataset = AutoBucketing(
                concepts_list=args.concepts_list,
                use_image_names_as_captions=args.use_image_names_as_captions,
                batch_size=args.train_batch_size,
                tokenizer=tokenizer,
                add_class_images_to_dataset=args.add_class_images_to_dataset,
                balance_datasets=args.auto_balance_concept_datasets,
                resolution=args.resolution,
                with_prior_loss=False,#args.with_prior_preservation,
                repeats=args.dataset_repeats,
                use_text_files_as_captions=args.use_text_files_as_captions,
                aspect_mode=args.aspect_mode,
                action_preference=args.aspect_mode_action_preference,
                seed=args.seed,
                model_variant=args.model_variant,
                extra_module=None if args.model_variant != "depth2img" else d2i,
                mask_prompts=args.mask_prompts,
            )
        else:
            train_dataset = NormalDataset(
            concepts_list=args.concepts_list,
            tokenizer=tokenizer,
            with_prior_preservation=args.with_prior_preservation,
            size=args.resolution,
            center_crop=args.center_crop,
            num_class_images=args.num_class_images,
            use_image_names_as_captions=args.use_image_names_as_captions,
            repeats=args.dataset_repeats,
            use_text_files_as_captions=args.use_text_files_as_captions,
            seed = args.seed,
            model_variant=args.model_variant,
            extra_module=None if args.model_variant != "depth2img" else d2i,
            mask_prompts=args.mask_prompts,
        )

    def collate_fn(examples):
        #print(examples)
        #print('test')
        input_ids = [example["instance_prompt_ids"] for example in examples]
        tokens = input_ids
        pixel_values = [example["instance_images"] for example in examples]
        if args.model_variant == 'inpainting':
            mask = [example["mask"] for example in examples]
        if args.model_variant == 'depth2img':
            depth = [example["instance_depth_images"] for example in examples]

        #print('test')
        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            if args.model_variant == 'inpainting':
                mask += [example["class_mask"] for example in examples]
            if args.model_variant == 'depth2img':
                depth = [example["class_depth_images"] for example in examples]
        if args.model_variant == 'inpainting':
            mask_values = torch.stack(mask)
            mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
        if args.model_variant == 'depth2img':
            depth_values = torch.stack(depth)
            depth_values = depth_values.to(memory_format=torch.contiguous_format).float()
        
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # Find the maximum length of the input_ids and clip to the next number of 75 tokens to avoid unruly SD front ends failing.
        max_len = max(len(x) for x in input_ids)

        # Calculate the number of chunks needed to process the input_ids in extended mode
        num_chunks = math.ceil(max_len / (tokenizer.model_max_length - 2))
        # Prevent zero dimensional tensors due to zero tokens
        if num_chunks < 1:
            num_chunks = 1

        # Trim our total token length into multiples of 75
        len_input = tokenizer.model_max_length - 2
        if num_chunks > 1:
            len_input = (tokenizer.model_max_length * num_chunks) - (num_chunks * 2)

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=len_input,
            return_tensors="pt",\
            ).input_ids

        if args.model_variant == 'base':
            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "extra_values": None,
                "tokens" : tokens
            }
        else:
            if args.model_variant == 'depth2img':
                extra_values = depth_values
            elif args.model_variant == 'inpainting':
                extra_values = mask_values
            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "extra_values": extra_values,
                "tokens" : tokens
            }
        return batch
    
    if not args.use_latents_only or args.regenerate_latent_cache:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True
        )
        if args.dataset_workers != -1:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, prefetch_factor=args.dataset_prefetch, num_workers=args.dataset_workers
            )
        #get the length of the dataset
        train_dataset_length = len(train_dataset)
        #code to check if latent cache needs to be resaved
        #check if last_run.json file exists in logging_dir
        if os.path.exists(logging_dir / "last_run.json"):
            #if it exists, load it
            with open(logging_dir / "last_run.json", "r") as f:
                last_run = json.load(f)
                last_run_batch_size = last_run["batch_size"]
                last_run_dataset_length = last_run["dataset_length"]
                if last_run_batch_size != args.train_batch_size:
                    print(f"{bcolors.WARNING}The batch_size has changed since the last run. Regenerating Latent Cache.{bcolors.ENDC}") 

                    args.regenerate_latent_cache = True
                    #save the new batch_size and dataset_length to last_run.json
                if last_run_dataset_length != train_dataset_length:
                    print(f"{bcolors.WARNING}The dataset length has changed since the last run. Regenerating Latent Cache.{bcolors.ENDC}") 

                    args.regenerate_latent_cache = True
                    #save the new batch_size and dataset_length to last_run.json
            with open(logging_dir / "last_run.json", "w") as f:
                json.dump({"batch_size": args.train_batch_size, "dataset_length": train_dataset_length}, f)
                    
        else:
            #if it doesn't exist, create it
            last_run = {"batch_size": args.train_batch_size, "dataset_length": train_dataset_length}
            #create the file
            with open(logging_dir / "last_run.json", "w") as f:
                json.dump(last_run, f)
    else:
        if os.path.exists(logging_dir / "last_run.json"):
            #if it exists, load it
            with open(logging_dir / "last_run.json", "r") as f:
                last_run = json.load(f)
                last_run_batch_size = last_run["batch_size"]
                if last_run_batch_size != args.train_batch_size:
                    print(f"{bcolors.WARNING}The batch_size has changed since the last run.{bcolors.ENDC}") 
                    raise Exception("Cannot comply, as mixed latent lengths may be present, please regenerate them.")
        else:
            raise Exception("Cannot comply, last_run.json not found.")

    
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "no":
        weight_dtype = torch.float32
    elif args.mixed_precision == "tf32":
        weight_dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        #torch.set_float32_matmul_precision("medium")

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema == True:
        ema_unet.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.model_variant == 'inpainting':
        if args.use_bucketing:
            wh = set([tuple(x.target_wh) for x in train_dataset.image_train_items])
        else:
            wh = set([tuple([args.resolution, args.resolution]) for x in train_dataset.image_paths])
        extra_latent = {shape: vae.encode(torch.zeros(1, 3, shape[1], shape[0]).to(accelerator.device, dtype=weight_dtype)).latent_dist.mean * 0.18215 for shape in wh}

    cached_dataset = CachedLatentsDataset(batch_size=args.train_batch_size,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    dtype=weight_dtype,
    model_variant=args.model_variant,
    shuffle_per_epoch=args.shuffle_per_epoch,
    args = args,
    accelerator=accelerator)
    if args.shuffle_per_epoch:
        print(f"{bcolors.WARNING}Will shuffle Latent Caches.{bcolors.ENDC}")

    gen_cache = False
    #data_len = len(train_dataloader)
    latent_cache_dir = Path(args.output_dir, "logs", "latent_cache")
        #check if latents_cache.pt exists in the output_dir
    if args.use_latents_only and not args.regenerate_latent_cache:
        if not os.path.exists(latent_cache_dir):
            raise Exception("Cannot load latents when the latents folder does not exist.")

        if len(os.listdir(latent_cache_dir)) == 0:
            raise Exception("Cannot load latents when there are no latent caches.")

        print(f"{bcolors.OKGREEN}Loading Latent Cache from {latent_cache_dir}{bcolors.ENDC}")
        del vae
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        data_len = len(os.listdir(latent_cache_dir))
        #load all the cached latents into a single dataset
        for i in os.listdir(latent_cache_dir):
            cached_dataset.add_pt_cache(os.path.join(latent_cache_dir, i))

    else:
        data_len = len(train_dataloader)
        if not os.path.exists(latent_cache_dir):
            os.makedirs(latent_cache_dir)
        for i in range(0,data_len-1):
            if not os.path.exists(os.path.join(latent_cache_dir, f"latents_cache_{i}.pt")):
                gen_cache = True
                break
        if args.regenerate_latent_cache:
                files = os.listdir(latent_cache_dir)
                gen_cache = True
                for file in files:
                    os.remove(os.path.join(latent_cache_dir,file))
        if gen_cache == False:
            print(f"{bcolors.OKGREEN}Loading Latent Cache from {latent_cache_dir}{bcolors.ENDC}")
            del vae

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            #load all the cached latents into a single dataset
            for i in range(0,data_len-1):
                cached_dataset.add_pt_cache(os.path.join(latent_cache_dir,f"latents_cache_{i}.pt"))
            
        if gen_cache == True:
            #delete all the cached latents if they exist to avoid problems
            print(f"{bcolors.WARNING}Generating latents cache...{bcolors.ENDC}")
            train_dataset = LatentsDataset([], [], [], [], [])
            counter = 0
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with torch.no_grad():
                for batch in tqdm(train_dataloader, desc="Caching latents", bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKBLUE, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,)):
                    cached_conditioning_latent = None
                    cached_extra = None
                    batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                    # if args.model_variant == "inpainting":
                    #     batch["extra_values"] = batch["extra_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    #     cached_conditioning_latent = vae.encode(batch["pixel_values"] * (1 - batch["extra_values"])).latent_dist
                    #     cached_extra = functional.resize(batch["extra_values"], size=cached_conditioning_latent.mean.shape[2:])
                    # if args.model_variant == "depth2img":
                    #     batch["extra_values"] = batch["extra_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    #     cached_conditioning_latent = vae.encode(batch["pixel_values"] * (1 - batch["extra_values"])).latent_dist
                    #     cached_extra = functional.resize(batch["extra_values"], size=cached_conditioning_latent.mean.shape[2:])
                    cached_latent = vae.encode(batch["pixel_values"]).latent_dist
                    cached_text_enc = batch["input_ids"]
                    train_dataset.add_latent(cached_latent, cached_text_enc, cached_conditioning_latent, cached_extra, batch["tokens"])
                    del batch
                    del cached_latent
                    del cached_text_enc
                    del cached_conditioning_latent
                    del cached_extra
                    torch.save(train_dataset, os.path.join(latent_cache_dir,f"latents_cache_{counter}.pt"))
                    cached_dataset.add_pt_cache(os.path.join(latent_cache_dir,f"latents_cache_{counter}.pt"))
                    counter += 1
                    train_dataset = LatentsDataset([], [], [], [], [])
                    #if counter % 300 == 0:
                        #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=False)
                    #    gc.collect()
                    #    torch.cuda.empty_cache()
                    #    accelerator.free_memory()

            #clear vram after caching latents
            del vae

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
   
    # Add unconditional batches
    if args.unconditional_dropout > 0:
        known_caches = cached_dataset.get_cache_list()
        # Only add more caches if there's more than 1 total latent batches
        if len(known_caches) > 2:
            additional_caches = random.sample(known_caches, int( (len(known_caches)-1) * args.unconditional_dropout))
            print(f"{bcolors.WARNING}Adding {len(additional_caches)} additional duplicate batches as unconditional guidance.{bcolors.ENDC}")
            for duplicate_cache in additional_caches:
                cached_dataset.add_pt_cache(duplicate_cache[0], dropout=True)
                data_len += 1
            del additional_caches
            del known_caches

    # Prepare the torch dataloader
    train_dataloader = torch.utils.data.DataLoader(cached_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=False)
    print(f"{bcolors.OKGREEN}Latents are ready.{bcolors.ENDC}")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    if args.lr_warmup_steps < 1:
        args.lr_warmup_steps = math.floor(args.lr_warmup_steps * args.max_train_steps / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.train_text_encoder and not args.use_ema:
        if args.using_fsdp:
            unet, text_encoder = accelerator.prepare(
                unet, text_encoder
            )
            optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
    elif args.train_text_encoder and args.use_ema:
        if args.using_fsdp:
            unet, text_encoder, ema_unet = accelerator.prepare(
                unet, text_encoder, ema_unet
            )
            optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, text_encoder, ema_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, ema_unet, optimizer, train_dataloader, lr_scheduler
            )
    elif not args.train_text_encoder and args.use_ema:
        if args.using_fsdp:
            unet, ema_unet = accelerator.prepare(
                unet, ema_unet
            )
            optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, ema_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, ema_unet, optimizer, train_dataloader, lr_scheduler
            )
    elif not args.train_text_encoder and not args.use_ema:
        if args.using_fsdp:
            unet = accelerator.prepare(
                unet
            )
            optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataloader)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")
    # Train!

    def save_and_sample_weights(step,context='checkpoint',save_model=True, auto_upload=False, complete=True):
        try:
            # Create the pipeline using using the trained modules and save it.
            if accelerator.is_main_process:
                if 'step' in context:
                    #what is the current epoch
                    epoch = step // num_update_steps_per_epoch
                else:
                    epoch = step
                if args.train_text_encoder and args.stop_text_encoder_training == True:
                    text_enc_model = accelerator.unwrap_model(text_encoder,True)
                elif args.train_text_encoder and args.stop_text_encoder_training > epoch:
                    text_enc_model = accelerator.unwrap_model(text_encoder,True)
                elif args.train_text_encoder == False:
                    text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
                elif args.train_text_encoder and args.stop_text_encoder_training <= epoch:
                    if 'frozen_directory' in locals():
                        text_enc_model = CLIPTextModel.from_pretrained(frozen_directory, subfolder="text_encoder")
                    else:
                        text_enc_model = accelerator.unwrap_model(text_encoder,True)
                    
                #schedule = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
                #schedule = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type="v_prediction")
                schedule = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                unwrapped_unet = accelerator.unwrap_model(unet,True)
                if args.use_ema:
                    ema_unet.copy_to(unwrapped_unet.parameters())
                    
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrapped_unet,
                    text_encoder=text_enc_model,
                    vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae",),
                    safety_checker=None,
                    torch_dtype=weight_dtype,
                    local_files_only=False,
                    requires_safety_checker=False
                )
                pipeline.scheduler = schedule
                save_name = f"{context}_{step+1}"
                if auto_upload:
                    if complete:
                        save_name = f"{args.project_name}_e{step}"
                    else:
                        save_name = f"{args.project_name}_e{step}-pre"

                    if args.project_append != "":
                        save_name += f"_{args.project_append}"

                save_dir = os.path.join(args.output_dir, save_name)
                if args.stop_text_encoder_training == True:
                    save_dir = frozen_directory
                if save_model:
                    pipeline.save_pretrained(save_dir, safe_serialization=True)
                    with open(os.path.join(save_dir, "args.json"), "w") as f:
                            json.dump(args.__dict__, f, indent=2)
                if args.stop_text_encoder_training == True:
                    #delete every folder in frozen_directory but the text encoder
                    for folder in os.listdir(save_dir):
                        if folder != "text_encoder" and os.path.isdir(os.path.join(save_dir, folder)):
                            shutil.rmtree(os.path.join(save_dir, folder))
                    del pipeline
                    del unwrapped_unet
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                if save_model == True:
                    tqdm.write(f"{bcolors.OKGREEN}Diffusers weights saved to {save_dir}{bcolors.ENDC}")
                    if auto_upload:
                        output_filename = f"{save_name}.safetensors"
                        save_tensors = os.path.join(args.output_dir, output_filename)
                        subprocess.run(["python", "scripts/convert_diffusers_to_sd_cli.py", save_dir, save_tensors])
        except Exception as e:
            print(e)
            print(f"{bcolors.FAIL}An Exeception occured during saving or uploading, skipping.{bcolors.ENDC}")
            pass

    # Only show the progress bar once on each machine.
    progress_bar_inter_epoch = tqdm(range(num_update_steps_per_epoch),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKGREEN, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process, desc="Steps to Epoch")
    progress_bar = tqdm(range(args.max_train_steps),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKBLUE, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process, desc="Overall Steps")
    progress_bar_e = tqdm(range(args.num_train_epochs),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKGREEN, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process, desc="Overall Epochs")

    global_step = 0
    loss_avg = AverageMeter()
    nat_loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()

    try:
        tqdm.write(f"{bcolors.OKBLUE}Starting Training!{bcolors.ENDC}")

        mid_generation = False
        mid_checkpoint = False
        mid_sample = False
        mid_checkpoint_step = False
        mid_sample_step = False
        mid_quit = False
        mid_quit_step = False
        #lambda set mid_generation to true
        frozen_directory=args.output_dir + "/frozen_text_encoder"

        # Get our limit of token chunks early.
        max_length = tokenizer.model_max_length
        max_standard_tokens = max_length - 2
        token_chunks_limit = math.ceil(args.token_limit / max_standard_tokens)
        if token_chunks_limit < 1:
            token_chunks_limit = 1

        
        get_noisy_latents = tu.get_noisy_latents
        get_embeddings = ""
        if args.train_text_encoder:
            get_embeddings = tu.text_encoder_training
        else:
            get_embeddings = tu.text_encoder_inference

        get_unet_noise = tu.predict_unet_noise
        get_loss = tu.get_batch_loss

        are_we_v_pred = False
        if args.force_v_pred:
            are_we_v_pred = True
        elif noise_scheduler.config.prediction_type == "v_prediction":
            are_we_v_pred = True

        for epoch in range(args.num_train_epochs):
            model_outputs = 0
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()
            
            if args.train_text_encoder and args.stop_text_encoder_training == epoch:
                args.stop_text_encoder_training = True
                if accelerator.is_main_process:
                    print(f"{bcolors.WARNING} Stopping text encoder training{bcolors.ENDC}")   
                    current_percentage = (epoch/args.num_train_epochs)*100
                    #round to the nearest whole number
                    current_percentage = round(current_percentage,0)

                    if os.path.exists(frozen_directory):
                        #delete the folder if it already exists
                        shutil.rmtree(frozen_directory)
                    os.mkdir(frozen_directory)
                    save_and_sample_weights(epoch,'epoch')
                    args.stop_text_encoder_training = epoch
            progress_bar_inter_epoch.reset(total=num_update_steps_per_epoch)
            e_steps = 0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet, text_encoder) if args.train_text_encoder and args.gradient_accumulation_steps > 1 else accelerator.accumulate(unet):
                    # Convert images to latent space
                    with torch.no_grad():
                        latent_dist = batch[0][0]
                        latents = latent_dist.sample() * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
                    timesteps = timesteps.long()

                    if args.with_pertubation_noise:
                        # https://arxiv.org/pdf/2301.11706.pdf
                        noisy_latents = noise_scheduler.add_noise(latents, noise + args.perturbation_noise_weight * torch.randn_like(latents), timesteps)
                    else:
                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents = noisy_latents.to(accelerator.device)

                    # Get the text embedding for conditioning
                    with text_enc_context:
                        tru_len = max(len(x) for x in batch[0][1])
                        max_chunks = np.ceil(tru_len / max_standard_tokens).astype(int)
                        max_len = max_chunks.item() * max_standard_tokens
                        clamp_event = False
                        #print(f"\n\n\nC:{max_chunks}, L:{max_len}, T:{tru_len}")
                        # If we're a properly padded bunch of tokens that have come from the tokeniser padder, train normally;
                        # otherwise we're handling a dropout batch, and thusly need to handle it the normal way. 
                        if tru_len == max_len and max_chunks > 1:
                            # Duplicate batch tensor to prevent irreparably damaging it's token data
                            # Recommended over torch.tensor()
                            n_batch = batch[0][1].clone().detach()
                            z = None
                            for i, x in enumerate(n_batch):
                                if len(x) < max_len:
                                    n_batch[i] = [*x, *np.full((max_len - len(x)), tokenizer.eos_token_id)]
                                del i
                                del x

                            chunks = [n_batch[:, i:i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)]
                            clamp_chunk = 0
                            for chunk in chunks:
                                # Hard limit the tokens to fit in memory for the rare event that latent caches that somehow exceed the limit.
                                if clamp_chunk > (token_chunks_limit):
                                    del chunk
                                    break

                                chunk = chunk.to(accelerator.device)
                                chunk = torch.cat((torch.full((chunk.shape[0], 1), tokenizer.bos_token_id).to(accelerator.device), chunk, torch.full((chunk.shape[0], 1), tokenizer.eos_token_id).to(accelerator.device)), 1)
                                encode = accelerator.unwrap_model(text_encoder)(chunk, output_hidden_states=True) if args.multi_gpu else text_encoder(chunk, output_hidden_states=True)

                                hidden_states = encode['hidden_states'][-2] if args.clip_penultimate else encode['hidden_states'][-1]
                                if args.using_fsdp:
                                    hidden_states = hidden_states.to(accelerator.device, dtype=torch.float32)
                                else:
                                    hidden_states = hidden_states.to(accelerator.device)

                                if z is None:
                                    z = accelerator.unwrap_model(text_encoder).text_model.final_layer_norm(hidden_states) if args.multi_gpu else text_encoder.text_model.final_layer_norm(hidden_states)
                                else:
                                    z = torch.cat((z, accelerator.unwrap_model(text_encoder).text_model.final_layer_norm(hidden_states)), dim=-2) if args.multi_gpu else torch.cat((z, text_encoder.text_model.final_layer_norm(hidden_states)), dim=-2)
                                        
                                del encode, hidden_states, chunk
                                clamp_chunk += 1
                            z = z.to(accelerator.device)
                            encoder_hidden_states = torch.stack(tuple(z))
                            encoder_hidden_states = encoder_hidden_states.to(accelerator.device)
                            del n_batch, tru_len, max_chunks, max_len, z, chunks, clamp_chunk, clamp_event

                        else:
                            if args.clip_penultimate == True:
                                encoder_hidden_states = text_encoder(batch[0][1], output_hidden_states=True)
                                encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
                            else:
                                encoder_hidden_states = text_encoder(batch[0][1])[0]
                            encoder_hidden_states = encoder_hidden_states.to(accelerator.device)

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon" and not args.force_v_pred:
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction" or args.force_v_pred:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.multi_gpu:
                        target = target.to(accelerator.device)
                        model_pred = model_pred.to(accelerator.device)

                    natural_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    natural_loss = natural_loss.mean([1, 2, 3])
                    loss = natural_loss.clone().detach()

                    if args.min_snr_gamma:
                        loss = tu.apply_snr_weight_neo(are_we_v_pred, natural_loss.float(), timesteps, noise_scheduler, args.min_snr_gamma, accelerator)
                    elif args.snr_debias:
                        loss = tu.snr_debias(are_we_v_pred, natural_loss.float(), timesteps, noise_scheduler, accelerator)

                    natural_loss = natural_loss.mean()
                    loss = loss.mean()
                    
                    del timesteps, noise, latents, noisy_latents, encoder_hidden_states
                    accelerator.backward(loss)

                    # Do not bother with clipping gradients if max_grad_norm is zero
                    if accelerator.sync_gradients and args.max_grad_norm > 0:
                        params_to_clip = (itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    loss_avg.update(loss.detach_(), bsz)
                    nat_loss_avg.update(natural_loss.detach_(), bsz)
                    del loss, natural_loss

                    if args.use_ema:
                        ema_unet.step(unet.parameters())

                if not global_step % args.log_interval:
                    if args.min_snr_gamma or args.snr_debias:
                        logs = {"loss": loss_avg.avg.item(), "nat loss": nat_loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                    else:
                        logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    del logs

                progress_bar.update(1)
                progress_bar_inter_epoch.update(1)
                progress_bar_e.refresh()
                global_step += 1
                e_steps += 1

                if e_steps == 400 and args.debug_flag:
                    try:
                        torch.cuda.memory._dump_snapshot(torch_mem_output)
                        torch.cuda.memory._record_memory_history(enabled=None)
                    except:
                        pass
                    
                # Clean up every 1% trained while under multi GPU mode while not in debug
                if args.multi_gpu and num_update_steps_per_epoch > 100:
                    if not e_steps % ((num_update_steps_per_epoch - 1) // 100):
                        if torch.cuda.is_available() and not args.debug_flag:
                            # Wait for processes to sync up then clear cache
                            accelerator.wait_for_everyone()
                            torch.cuda.empty_cache()
                            accelerator.free_memory()
                            gc.collect()
                            # And do so afterwards to prevent any losses during training
                            accelerator.wait_for_everyone()

                if args.half_completion_upload and not args.save_every_quarter:
                    if not e_steps % (num_update_steps_per_epoch // 2):
                        if e_steps > 0 and e_steps < num_update_steps_per_epoch-1:
                            if args.webhook_url != "":
                                # Use the regular epoch value for normal training runs, not when supplied via argument.
                                if args.num_train_epochs == 1:
                                    accelerator.wait_for_everyone()
                                    save_and_sample_weights(int(args.epoch_number), 'epoch', save_model=True, auto_upload=True, complete=False)
                                elif args.num_train_epochs > 1 and args.num_train_epochs > int(args.epoch_number):
                                    accelerator.wait_for_everyone()
                                    save_and_sample_weights(epoch, 'epoch', save_model=True, auto_upload=True, complete=False)
                elif args.save_every_quarter:
                    if not e_steps % (num_update_steps_per_epoch // 4):
                        if e_steps > 0 and model_outputs < 3:
                            accelerator.wait_for_everyone()
                            save_and_sample_weights(global_step,'step',save_model=True)
                            model_outputs += 1

                if global_step >= args.max_train_steps:
                    break
            progress_bar_e.update(1)
            if not epoch % args.save_every_n_epoch:
                accelerator.wait_for_everyone()
                save_and_sample_weights(epoch,'epoch')
            if args.seed is not None and args.epoch_seed:
                set_seed(args.seed + (1 + epoch))
            accelerator.wait_for_everyone()
    except Exception as e:
        print("Something went tits up.")
        raise e
    except KeyboardInterrupt:
        print("SIGINT/CTRL + C detected, stopping.")
    # If it's being saved every epoch, we don't need to save another
    if args.save_every_n_epoch != 1:
        save_and_sample_weights(args.num_train_epochs,'epoch')

    accelerator.end_training()

if __name__ == "__main__":
    main()
