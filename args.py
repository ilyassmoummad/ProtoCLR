import argparse

parser = argparse.ArgumentParser()

# Generic
parser.add_argument("--report", action='store_true') #write results in a txt file
parser.add_argument("--save", action='store_true') #save model checkpoints
parser.add_argument("--debug", action='store_true') #load data only, to make sure it is correctly loaded
parser.add_argument("--device", type=str, default='cuda') #device to use ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...]

# Path
parser.add_argument("--traindir", type=str, default='') #path for the training data
parser.add_argument("--evaldir", type=str, default='') #path for the evaluation (validation/testing) data
parser.add_argument("--modelpath", type=str, default='') #path where to save checkpoints
parser.add_argument("--modelckpt", type=str, default='') #path of the checkpoint (.pth)

# Data Processing
parser.add_argument("--sr", type=int, default=16000) #samplerate
parser.add_argument("--duration", type=int, default=5) #duration in seconds of audios
parser.add_argument("--maxevents", type=int, default=1) #max number of events per audio file
parser.add_argument("--nworkers", type=int, default=16) #number of workers

# Mel Spectrogram
parser.add_argument("--nmels", type=int, default=128) #number of mels
parser.add_argument("--nfft", type=int, default=1024) #size of FFT
parser.add_argument("--hoplen", type=int, default=320) #hop between STFT windows
parser.add_argument("--fmax", type=int, default=8000) #fmax
parser.add_argument("--fmin", type=int, default=50) #fmin

# Data Augmentation
parser.add_argument("--mincoef", type=float, default=0.6) #minimum coef for spectrogram mixing
parser.add_argument("--fmask", type=int, default=10) #fmax
parser.add_argument("--tmask", type=int, default=30) #tmax
parser.add_argument("--fstripe", type=int, default=3) #fstripe
parser.add_argument("--tstripe", type=int, default=5) #tstripe

# Loss
parser.add_argument("--loss", type=str, default='protoclr') #loss to use for training ['simclr', 'supcon', 'protoclr', 'ce']
parser.add_argument("--tau", type=float, default=0.1) #temperature for cosine sim

# Training
parser.add_argument("--adam", action='store_true') #use adam instead of sgd
parser.add_argument("--bs", type=int, default=256) #batch size for representation learning
parser.add_argument("--epochs", type=int, default=100) #nb of epochs to train the feature extractor on the training set
parser.add_argument("--savefreq", type=int, default=100) #nb of epochs between each model save
parser.add_argument("--lr", type=float, default=5e-2) #learning rate for pretraining
parser.add_argument("--momentum", type=float, default=0.9) #sgd momentum
parser.add_argument("--wd", type=float, default=1e-6) #weight decay
parser.add_argument("--outdim", type=int, default=128) #output dimension of projector

# Validation
parser.add_argument("--k", type=int, default=1) #k for K Nearest Neighbor

# Few-Shot Evaluation
parser.add_argument("--nruns", type=int, default=10) #number of few shot runs
parser.add_argument("--nshots", type=int, default=1) #K for number of shots per class

# Testing
parser.add_argument("--testds", type=str, default='') #testing dataset ['HSN', 'NBP', 'NES', 'PER', 'SNE', 'SSW', 'UHH'] and val dataset 'POW'

args = parser.parse_args()