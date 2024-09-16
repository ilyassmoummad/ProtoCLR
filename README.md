# Domain-Invariant Representation Learning of Bird Sounds

## Authors
<sup>1</sup> Ilyass Moummad, <sup>2</sup> Romain Serizel, <sup>3</sup> Emmanouil Benetos, <sup>1</sup> Nicolas Farrugia

<sup>1</sup> IMT Atlantique, Lab-STICC, Brest, France

<sup>2</sup> Université de Lorraine, LORIA, INRIA, Nancy, France  

<sup>3</sup> C4DM, Queen Mary University of London, London, UK  

## ProtoCLR (Prototypical Contrastive Learning of Representations)
In this work, we propose ProtoCLR as a more efficient alternative to SupCon. Both methods demonstrate robustness to domain shift, specifically when pre-training on focal recordings and testing on soundscape recordings. This repository provides the code to train feature extractors on Xeno-Canto and perform few-shot evaluation on various downstream tasks.

## Preprint
[Domain-Invariant Representation Learning of Bird Sounds](https://arxiv.org/abs/2409.08589)

## Getting Started

### Step 1: BirdSet Installation
1. Clone the [BirdSet](https://github.com/DBD-research-group/BirdSet) benchmark repository.
2. Follow the installation steps in their repo and download the required datasets.

### Step 2: Install Dependencies
Install requirements.txt from both BirdSet and this repository:

```bash
pip install -r requirements.txt 
```

### Step 3 (Optional): Sanity Check for Dataset Loading

```bash
python3 train_encoder.py --debug
```

### Step 4: Train the Encoder
**ProtoCLR with CvT-13**

```bash
python3 train_encoder.py --loss protoclr --pretrainds XCM --epochs 100 --adam --nworkers 16 --bs 256 --lr 5e-4 --wd 1e-6 --device cuda:0 --model cvt13 --datadir birdset_path --savepath path_for_checkpoint --save
```

**SupCon with MobileNetV3-Large**

```bash
python3 train_encoder.py --loss supcon --pretrainds XCM --epochs 100 --nworkers 16 --bs 1024 --lr 5e-2 --wd 1e-6 --device cuda:0 --model mobilenetv3l --datadir birdset_path --savepath path_for_checkpoint --save
```

For more details about the arguments, refer to args.py.

**Note** Adjust the number of workers (--nworkers) based on your machine to avoid data loader bottlenecks, which can slow down training.

### Step 5: Few-shot Evaluation

To evaluate the model on a 5-shot task, run the following:
```bash
python3 test_fewshot.py --nshots 5 --model mobilenetv3l --modelckpt /path/to/mobilenetv3l_supcon_pretrain_XCM_checkpoint.pth --bs 1024 --nworkers 16 --datadir /path/to/hf_birdset/
```
**Note** Reduce the batch size (--bs) if it doesn't fit in your GPU memory.

## More details and checkpoints (soon)

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