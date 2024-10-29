from utils import Normalization, Standardization, Projector, model_dim
from augmentations import TimeShift, MixRandom, SpecAugment
from losses import SupConLoss, ProtoCLRLoss
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from dataset import PTDataset
from train_utils import train
from args import args
import torch.nn as nn
from cvt import cvt13
import warnings
import torch
import os

warnings.filterwarnings("ignore")

MEAN, STD = 0.5347, 0.0772 # mean and std of XC

train_ds = PTDataset(args.traindir)
N_CLASSES = len(train_ds.class_to_idx)

train_loader = DataLoader(dataset=train_ds, batch_size=args.bs, num_workers=args.nworkers, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=True)
val_ds = torch.load(os.path.join(args.evaldir, 'pow.pt'))

# Preprocessing Transformations
time_steps = 301 # int(args.sr*args.duration/args.hoplen)=300
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
encoder = cvt13().to(args.device) # CvT-13
model_name = 'cvt'

# Loss
if args.loss in ['simclr', 'supcon']:
    loss_fn = SupConLoss(tau=args.tau)
elif args.loss == 'protoclr':
    loss_fn = ProtoCLRLoss(tau=args.tau)
elif args.loss == 'ce':
    loss_fn = nn.CrossEntropyLoss()

# Projector / Classifier
if args.loss != 'ce':
    projector = Projector(model_name=model_name, out_dim=args.outdim).to(args.device)
else:
    projector = nn.Linear(model_dim[model_name], N_CLASSES).to(args.device)

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
    encoder_state_dict, projector_state_dict = train(encoder, projector, train_loader, val_ds, train_transform, val_transform, loss_fn, optim, args) 
    # Saving CKPT
    if args.save:   
        os.makedirs(args.modelpath, exist_ok=True) 
        best_model_path = os.path.join(args.modelpath, args.loss + '_pretrain_' + '_epochs' + str(args.epochs) + '_lr' + str(args.lr) + '_trial1.pth')
        if os.path.isfile(best_model_path):
            i = 1
            while os.path.isfile(best_model_path):
                i += 1
                best_model_path = best_model_path[:-5] + str(i) + '.pth'

        best_model_path = best_model_path.replace('.pth', '_best.pth')

        torch.save({'encoder': encoder_state_dict, 'classifier': projector_state_dict}, best_model_path)
else:
    last_state_dict = train(encoder, projector, train_loader, val_ds, train_transform, val_transform, loss_fn, optim, args)
    # Saving CKPT
    if args.save:   
        os.makedirs(args.modelpath, exist_ok=True) 
        last_model_path = os.path.join(args.modelpath, args.loss + '_pretrain_' + '_epochs' + str(args.epochs) + '_lr' + str(args.lr) + '_trial1.pth')
        if os.path.isfile(last_model_path):
            i = 1
            while os.path.isfile(last_model_path):
                i += 1
                last_model_path = last_model_path[:-5] + str(i) + '.pth'

        last_model_path = last_model_path.replace('.pth', '_last.pth')

        torch.save(last_state_dict, last_model_path)