# Domain-Invariant Representation Learning of Bird Sounds

## Authors
<sup>1</sup> Ilyass Moummad, <sup>2</sup> Romain Serizel, <sup>3</sup> Emmanouil Benetos, <sup>1</sup> Nicolas Farrugia

<sup>1</sup> IMT Atlantique, Lab-STICC, Brest, France  
<sup>2</sup> Université de Lorraine, LORIA, INRIA, Nancy, France  
<sup>3</sup> C4DM, Queen Mary University of London, London, UK  

---

## Overview
This repository introduces [ProtoCLR](https://arxiv.org/abs/2409.08589), a Prototypical Contrastive Learning approach designed for robust representation learning. ProtoCLR has been validated on transfer learning tasks for bird sound classification, showing strong domain-invariance in few-shot scenarios.

Our approach uses focal recordings for pre-training and soundscape recordings for evaluation, demonstrating robustness to domain shifts. The results provided in this repository are based on models trained for 100 epochs, while ongoing experiments with 300 epochs are being added to explore extended training benefits.

## Preprint
Read the full paper: [Domain-Invariant Representation Learning of Bird Sounds](https://arxiv.org/abs/2409.08589)

---

## Checkpoints
The pre-trained ProtoCLR model checkpoint is available on Hugging Face and can be directly accessed and integrated into your bioacoustic projects.

- [ProtoCLR CvT-13 300 epochs model checkpoint](https://huggingface.co/ilyassmoummad/ProtoCLR)

### Audio Preparation Guidelines
To use the model effectively, ensure your audio files meet the following criteria:
- **Sample Rate**: 16 kHz
- **Padding**: For audio shorter than 6 seconds, pad with zeros or repeat to reach a 6-second length.
- **Chunking**: For audio longer than 6 seconds, split into 6-second chunks.

### Example: Loading, Processing, and Running Inference on an Audio File
This example demonstrates how to load an audio file, preprocess it, and run inference with the ProtoCLR model.

```python
import torch
from cvt import cvt13  # Import model architecture
from melspectrogram import MelSpectrogramProcessor  # Import Mel spectrogram processor

# Initialize preprocessor and model
preprocessor = MelSpectrogramProcessor()
model = cvt13()
model.load_state_dict(torch.load("protoclr.pth"))
model.eval()

# Load and preprocess a sample audio waveform
def load_waveform(file_path):
    # Replace with audio loading code, e.g., torchaudio to load and resample
    pass

waveform = load_waveform("path/to/audio.wav")  # Load audio here

# Ensure waveform is sampled at 16 kHz, then pad/chunk to reach a 6-second length
input_tensor = preprocessor.process(waveform).unsqueeze(0)  # Add batch dimension

# Run the model on preprocessed audio
with torch.no_grad():
    output = model(input_tensor)
    print("Model output shape:", output.shape)
```

---

## Experiments

### Step 1: Downloading Data

We use datasets from the information retrieval benchmark [BIRB](https://arxiv.org/abs/2312.07439) and adapt them for few-shot learning, preparing them in `.pt` format with Xeno-Canto as the pretraining dataset, which includes focal recordings of over 10,000 bird species. For evaluation, various downstream soundscape datasets are provided, each featuring 6-second audio segments selected for peak bird activation using CNN14 from PANNs, with all recordings downsampled to 16kHz. This lightweight data format simplifies the training and evaluation of deep neural networks for bird sound classification, making it especially suited for few-shot learning.

#### 1.1 Training Data
The Xeno-Canto dataset contains a large collection of bird sound recordings optimized for few-shot learning tasks in the context of bird species classification.

- **Dataset Summary**:
  - **Total Examples**: 684,744 audio segments
  - **Segment Length**: Each segment is 6 seconds long
  - **Sampling Rate**: 16kHz
  - **Classes**: 10,127 unique bird species, each represented by its eBird species code

1. Download Xeno-Canto training data using this [script from Hugging Face](https://huggingface.co/datasets/ilyassmoummad/Xeno-Canto-6s-16khz/blob/main/download.py). Make sure to set the variable `DESTINATION_PATH` to your desired download location.

2. Merge and decompress the downloaded tar files:
   ```bash
   cat DESTINATION_PATH/*tar* > DESTINATION_PATH/xc-6s-16khz-pann.tar
   tar -xvf DESTINATION_PATH/xc-6s-16khz-pann.tar -C DATASET_PATH

#### 1.2 Evaluation Data (Validation and Test)
The evaluation datasets are sourced from the BIRB benchmark, a collection of soundscape datasets for evaluating bird sound classification in challenging, real-world conditions.

Download the evaluation datasets (validation and test) from [Zenodo](https://zenodo.org/records/13994373).

- **Validation Dataset**:
  - File: `pow.pt`
  - Contains 16,047 examples across 43 classes. Organized as a dictionary with `data` and `label` keys to enable efficient and rapid loading during validation. Classes with only one example are omitted to support one-shot classification tasks.

- **Test Datasets**:
  Each test dataset is in `.pt` files organized by species codes within subfolders. Use the metadata file from the training set to map species codes to common names.

  | Dataset | Subfolder | Examples | Classes | Zenodo Source |
  | ------- | --------- | -------- | ------- | ------------- |
  | SSW     | `ssw/`    | 50,760   | 96      | [Link](https://zenodo.org/records/7079380#.Y7ijHOxudhE) |
  | NES     | `coffee_farms/` | 6,952 | 89 | [Link](https://zenodo.org/records/7525349#.ZB8z_-xudhE) |
  | UHH     | `hawaii/` | 59,583   | 27      | [Link](https://zenodo.org/records/7078499#.Y7ijPuxudhE) |
  | HSN     | `high_sierras/` | 10,296 | 19 | [Link](https://zenodo.org/records/7525805#.ZB8zsexudhE) |
  | SNE     | `sierras_kahl/` | 20,147 | 56 | [Link](https://zenodo.org/records/7050014#.Y7ijWexudhE) |
  | PER     | `peru/`   | 14,768   | 132     | [Link](https://zenodo.org/records/7079124#.Y7iis-xudhE) |

### Step 2: Install Dependencies
Install requirements.txt from this repository:

```bash
pip install -r requirements.txt 
```

### Step 3: Train Feature Extractor
**ProtoCLR**

```bash
python3 train_encoder.py --loss protoclr --epochs 300 --nworkers 16 --bs 256 --lr 5e-4 --wd 1e-6 --device cuda:0 --traindir Path_to_Xeno-Canto-6s-16khz/ --evaldir Path_to_parent_folder_of_pow.pt --save --savefreq --freq 100
```

- **`--loss`**: Specifies the loss function to use. The following losses are supported: `protoclr`, `supcon`, `simclr`, and `ce` (cross-entropy).

- **`--traindir`**: Path to the training data directory containing the Xeno-Canto bird sound dataset. This directory should contain the decompressed data downloaded from Hugging Face.

- **`--evaldir`**: Path to the evaluation data directory where the validation file (`pow.pt`) is stored. This file will be used for evaluating model performance during training.

For more details about the arguments, refer to args.py.

**Note** Adjust the number of workers (--nworkers) based on your machine to avoid data loader bottlenecks which can slow down training.

### Step 4: Few-shot Evaluation

To evaluate the model on one- and five-shot tasks, run the following script:
```bash
python3 test_fewshot.py --modelckpt /path/to/weights.pth --bs 1024 --nworkers 16 --evaldir /path/to/soundscapes --report
```
**Note** Reduce the batch size (--bs) if it doesn't fit in your GPU memory.

---

## Citation
```
@misc{moummad2024dirlbs,
      title={Domain-Invariant Representation Learning of Bird Sounds}, 
      author={Ilyass Moummad and Romain Serizel and Emmanouil Benetos and Nicolas Farrugia},
      year={2024},
      eprint={2409.08589},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2409.08589}, 
}
```