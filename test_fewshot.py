from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.configs import LoaderConfig, LoadersConfig
from birdset.datamodule.components import XCEventMapping, EventDecoding
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.datamodule.base_datamodule import DatasetConfig

from utils import Normalization, Standardization, model_dim
from test_utils import sample_episode, nearest_prototype
from mobilenetv3 import mobilenetv3s, mobilenetv3l
from torchaudio import transforms as T
from test_utils import fewshoteval
from tqdm import tqdm
import torch.nn as nn
from args import args
from cvt import cvt13
import warnings
import torch
import os

warnings.filterwarnings("ignore")

MEAN, STD = 0.5347, 0.0772

report = {}

modelcheckpoint = args.modelckpt

print(f"{args.nshots}-shot Learning")

for testds in ['POW', 'PER', 'NES', 'UHH', 'HSN', 'NBP', 'SSW', 'SNE']:

    data_config = DatasetConfig(
        data_dir=os.path.join(args.datadir, testds),
        dataset_name=testds,
        hf_path='DBD-research-group/BirdSet',
        hf_name=testds,
        n_workers=args.nworkers,
        val_split=0.2,
        task="multiclass",
        classlimit=500,
        eventlimit=5,
        sampling_rate=args.sr,
    )

    fs_loader_config = LoaderConfig(
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.nworkers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    loaders_config = LoadersConfig(
        train=fs_loader_config,
        valid=fs_loader_config,
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

    fs_ds = BirdSetDataModule(
        dataset = data_config,
        loaders = loaders_config,
        transforms = transforms,
    )

    fs_ds.prepare_data()

    fs_ds.setup(stage="fit")

    loader1 = fs_ds.train_dataloader()
    loader2 = fs_ds.val_dataloader()

    time_steps = 251 # int(args.sr*args.duration/args.hoplen)=250
    melspec = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels)
    power_to_db = T.AmplitudeToDB()
    norm = Normalization()
    sd = Standardization(mean=MEAN, std=STD)
    transform = nn.Sequential(melspec, power_to_db, norm, sd).to(args.device)

    if args.model == 'mobilenetv3s':
        encoder = mobilenetv3s()
    elif args.model == 'mobilenetv3l':
        encoder = mobilenetv3l()
    elif args.model == 'cvt13':
        encoder = cvt13()
    else:
        print("Invalid model")
        quit()

    avgacc = []
    avgstd = []

    if 'ce' in modelcheckpoint:
        encoder.load_state_dict(torch.load(modelcheckpoint, map_location='cpu')['encoder'])
    else:
        encoder.load_state_dict(torch.load(modelcheckpoint, map_location='cpu'))
    encoder = encoder.to(args.device)

    print(f"Few-shot eval on {testds}")
    acc, std = fewshoteval(encoder, loader1, loader2, transform, args)
    print(f"Acc for {args.nruns} runs on {testds}: {acc}+-{std}")
    report[testds] = {'acc': acc, 'std': std}

print("---------------------------")
print(f"Results for {shot}-shot:\n")
for testds in ['POW', 'PER', 'NES', 'UHH', 'HSN', 'NBP', 'SSW', 'SNE']:
    print(f"{testds}: {report[testds]['acc']}+-{report[testds]['std']}\n")