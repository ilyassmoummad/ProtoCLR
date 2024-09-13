from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.configs import LoaderConfig, LoadersConfig
from birdset.datamodule.components import XCEventMapping, EventDecoding
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.datamodule.base_datamodule import DatasetConfig

from utils import Normalization, Standardization, Projector, model_dim
from augmentations import TimeShift, MixRandom, SpecAugment
from losses import SupConLoss, ProtoCLRLoss
from torchaudio import transforms as T
from mobilenetv3 import mobilenetv3l, mobilenetv3s
from cvt import cvt13
from train_utils import train
from val_utils import val_knn
import torch.nn as nn
import torch

import warnings
warnings.filterwarnings("ignore")

import os

from args import args

MEAN, STD = 0.5347, 0.0772 # mean and std of XC

if args.pretrainds == 'XCL':
    N_CLASSES = 9736 # for XCL
elif args.pretrainds == 'XCM':
    N_CLASSES = 411 # for XCM
else:
    print("Invalid pretrainds, select XCM or XCL")
    quit()

# Pretrain Config
xc_config = DatasetConfig(
    data_dir=os.path.join(args.datadir, args.pretrainds),
    dataset_name=args.pretrainds,
    hf_path='DBD-research-group/BirdSet',
    hf_name=args.pretrainds,
    n_workers=args.nworkers,
    val_split=0.05,
    task="multiclass",
    subset=None, #500 <- debug with a subset
    classlimit=500,
    eventlimit=args.maxevents,
    sampling_rate=args.sr,
)

# Configuration for the training data loader
train_loader_config = LoaderConfig(
    batch_size=args.bs,
    shuffle=True,
    num_workers=args.nworkers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

# Aggregating the loader configurations
loaders_config = LoadersConfig(
    train=train_loader_config,
)

transforms = BirdSetTransformsWrapper(
    task="multiclass",
    sampling_rate=args.sr,
    model_type="waveform",
    decoding=None,
    feature_extractor=None,
    max_length=args.duration,
    nocall_sampler=None,
    preprocessing=None,
)

xc_ds = BirdSetDataModule(
    dataset = xc_config,
    loaders = loaders_config,
    transforms = transforms,
)

xc_ds.prepare_data() # download or load dataset

xc_ds.setup(stage="fit") # required to get train and val loaders

train_loader = xc_ds.train_dataloader() # 1004 * 256
print(f"len of train set: {len(xc_ds.train_dataset)}")

###########################

# Val Config
pow_config = DatasetConfig(
    data_dir=os.path.join(args.datadir, 'POW'),
    dataset_name='POW',
    hf_path='DBD-research-group/BirdSet',
    hf_name='POW',
    n_workers=args.nworkers,
    val_split=0.2,
    task="multiclass",
    classlimit=500,
    eventlimit=args.maxevents,
    sampling_rate=args.sr,
)

# Configuration for the training data loader
train_loader_config = LoaderConfig(
    batch_size=args.bs,
    shuffle=True,
    num_workers=args.nworkers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

# Configuration for the testing data loader
test_loader_config = LoaderConfig(
    batch_size=args.bs,
    shuffle=False,
    num_workers=args.nworkers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
)

# Aggregating the loader configurations
loaders_config = LoadersConfig(
    train=train_loader_config,
    valid=test_loader_config,
)

transforms = BirdSetTransformsWrapper(
    task="multiclass",
    sampling_rate=args.sr,
    model_type="waveform",
    decoding=None,
    feature_extractor=None,
    max_length=args.duration,
    nocall_sampler=None,
    preprocessing=None,
)

pow_ds = BirdSetDataModule(
    dataset = pow_config,
    loaders = loaders_config,
    transforms = transforms,
)

pow_ds.prepare_data() # download or load dataset

pow_ds.setup(stage="fit") # required to get train and val loaders
knn_train_loader = pow_ds.train_dataloader()
knn_val_loader = pow_ds.val_dataloader()

if args.debug:
    print("All loaders have been loaded successfully.")
    quit()

# Preprocessing Transformations
time_steps = 251 # int(args.sr*args.duration/args.hoplen)=250
melspec = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels)
power_to_db = T.AmplitudeToDB()
norm = Normalization()
sd = Standardization(mean=MEAN, std=STD)

# Data Augmentations
mix = MixRandom(min_coef=args.mincoef)
tshift = TimeShift(Tshift=time_steps)
mask = SpecAugment(freq_mask=args.fmask, time_mask=args.tmask, freq_stripes=args.fstripe, time_stripes=args.tstripe)
if args.loss == 'ce':
    mix = nn.Identity()
train_transform = nn.Sequential(melspec, power_to_db, norm, tshift, mix, mask, sd).to(args.device)
val_transform = nn.Sequential(melspec, power_to_db, norm, sd).to(args.device)

# Model
if args.model == 'mobilenetv3s':
    encoder = mobilenetv3s().to(args.device)
elif args.model == 'mobilenetv3l':
    encoder = mobilenetv3l().to(args.device)
elif args.model == 'cvt13':
    encoder = cvt13().to(args.device)
else:
    print("Invalid model")
    quit()

# Loss
if args.loss in ['simclr', 'supcon']:
    loss_fn = SupConLoss(tau=args.tau)
elif args.loss == 'ce':
    loss_fn = nn.CrossEntropyLoss()

# Projector / Classifier
if args.loss != 'ce':
    projector = Projector(model_name=args.model, out_dim=args.outdim).to(args.device)
else:
    projector = nn.Linear(model_dim[args.model], N_CLASSES).to(args.device)

# Optimizer
trainable_parameters = list(encoder.parameters()) + list(projector.parameters())
if args.adam:
    optim = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.wd)
else:
    optim = torch.optim.SGD(trainable_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

num_encoder_params = sum(p.numel() for p in encoder.parameters())
total_params = num_encoder_params + sum(p.numel() for p in projector.parameters())
print(f"Total num of params: {round(total_params * 1e-6, 2)}M, encoder params: {round(num_encoder_params * 1e-6, 2)}M.")

# Training
if args.loss == 'ce':
    encoder_state_dict, projector_state_dict = train(encoder, projector, train_loader, knn_train_loader, knn_val_loader, pow_ds.len_trainset, train_transform, val_transform, loss_fn, optim, args) 

    # Saving CKPT
    if args.save:   
        os.makedirs(args.savepath, exist_ok=True) 
        best_model_path = os.path.join(args.savepath, args.model + '_' + args.loss + '_pretrain_' + args.pretrainds + '_epochs' + str(args.epochs) + '_lr' + str(args.lr) + '_trial1.pth')
        if os.path.isfile(best_model_path):
            i = 1
            while os.path.isfile(best_model_path):
                i += 1
                best_model_path = best_model_path[:-5] + str(i) + '.pth'

        torch.save({'encoder': encoder_state_dict, 'classifier': projector_state_dict}, best_model_path)
else:
    last_state_dict = train(encoder, projector, train_loader, knn_train_loader, knn_val_loader, pow_ds.len_trainset, train_transform, val_transform, loss_fn, optim, args)

    # Saving CKPT
    if args.save:   
        os.makedirs(args.savepath, exist_ok=True) 
        last_model_path = os.path.join(args.savepath, args.model + '_' + args.loss + '_pretrain_' + args.pretrainds + '_epochs' + str(args.epochs) + '_trial1.pth')
        if os.path.isfile(last_model_path):
            i = 1
            while os.path.isfile(last_model_path):
                i += 1
                last_model_path = last_model_path[:-5] + str(i) + '.pth'

        torch.save(last_state_dict, last_model_path)