# Domain-Invariant Representation Learning of Bird Sounds

## Authors
<sup>1</sup> Ilyass Moummad, <sup>2</sup> Romain Serizel, <sup>3</sup> Emmanouil Benetos, <sup>1</sup> Nicolas Farrugia

<sup>1</sup> IMT Atlantique, Lab-STICC, Brest, France

<sup>2</sup> Université de Lorraine, LORIA, INRIA, Nancy, France  

<sup>3</sup> C4DM, Queen Mary University of London, London, UK  

## ProtoCLR (Prototypical Contrastive Learning of Representations)

In this work, we introduce [ProtoCLR](https://arxiv.org/abs/2409.08589), an efficient alternative to [SupCon](https://arxiv.org/abs/2004.11362), specifically designed for representation learning. We validate ProtoCLR in transfer learning for bird sound classification, pre-training on focal recordings and evaluating on soundscape recordings in few-shot scenarios. Our approach demonstrates strong robustness to domain shift, making it well-suited for real-world applications.

The results reported in the pre-print are based on experiments conducted over 100 epochs. Current experiments are being extended to 300 epochs to explore the benefits of longer training in contrastive learning, which will be included in the upcoming days/weeks. Additionally, we enhance ProtoCLR by incorporating a hierarchical structure, enabling it to capture finer-grained relationships among bird species.

This repository provides detailed instructions for downloading and preparing training and evaluation datasets, hosted on Hugging Face and Zenodo in `.pt` format. The straightforward format and provided code ensure easy accessibility and reproducibility, supporting research in bird sound classification, especially within few-shot learning contexts.

## Preprint
[Domain-Invariant Representation Learning of Bird Sounds](https://arxiv.org/abs/2409.08589)

## Getting Started

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

## Checkpoints

The pre-trained ProtoCLR model checkpoint is now available on Hugging Face and can be directly accessed for use in your bioacoustic projects with few lines of code.

- [ProtoCLR CvT-13 300 epochs model checkpoint](https://huggingface.co/ilyassmoummad/ProtoCLR)

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