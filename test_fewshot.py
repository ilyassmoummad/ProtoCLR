from utils import Normalization, Standardization
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from eval_utils import fewshot_test
from dataset import PTDataset
from cvt import cvt13
from args import args
import torch.nn as nn
import warnings
import torch
import os

warnings.filterwarnings("ignore")

modelckpt = args.modelckpt
FILENAME = "./results_" + modelckpt.split('/')[-1][:-4] + ".txt"

MEAN, STD = 0.5347, 0.0772

report = {}

if args.report:
    with open(FILENAME, 'a+') as file:
        text = f"{modelckpt}"
        file.write(text + '\n')
for shot in [1, 5]:
    print(f"{shot}-shot")
    if args.report:
        with open(FILENAME, 'a+') as file:
            text = f"{shot}-shot"
            file.write(text + '\n')
    for testds in ['PER', 'NES', 'UHH', 'HSN', 'SSW', 'SNE']:
        if testds == 'PER':
            testds_name = 'peru'
        elif testds == 'NES':
            testds_name = 'coffee_farms'
        elif testds == 'UHH':
            testds_name = 'hawaii'
        elif testds == 'HSN':
            testds_name = 'high_sierras'
        elif testds == 'SSW':
            testds_name = 'ssw'
        elif testds == 'SNE':
            testds_name = 'sierras_kahl'

        fs_ds = PTDataset(os.path.join(args.evaldir, testds_name))
        loader = DataLoader(dataset=fs_ds, batch_size=args.bs, num_workers=args.nworkers, persistent_workers=True, pin_memory=True, shuffle=False, drop_last=False)

        # Preprocessing Transformations
        time_steps = 301 # int(args.sr*args.duration/args.hoplen)=300
        melspec = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels)
        power_to_db = T.AmplitudeToDB()
        norm = Normalization()
        sd = Standardization(mean=MEAN, std=STD)
        transform = nn.Sequential(melspec, power_to_db, norm, sd).to(args.device)

        # Model
        encoder = cvt13()#.to(args.device)
        if 'ce' in modelckpt:
            encoder.load_state_dict(torch.load(modelckpt, map_location='cpu')['encoder']) # my code saves both encoder and projector for cross-entropy
        else:
            encoder.load_state_dict(torch.load(modelckpt, map_location='cpu'))
        encoder = encoder.to(args.device)

        # Testing
        acc, std = fewshot_test(encoder, loader, transform, args, shot)

        report[testds] = {'acc': acc, 'std': std}
        print(f"{testds}: {report[testds]['acc']}+-{report[testds]['std']}\n")

    if args.report:
        with open(FILENAME, 'a+') as f:
            for testds in ['PER', 'NES', 'UHH', 'HSN', 'SSW', 'SNE']:
                f.write(f"{testds}: {report[testds]['acc']}+-{report[testds]['std']}\n")