# Artery–Vein Segmentation

Attention-WNet and Swin-WNet research implementations for retinal images.

## Runnable deep-supervision project

[`deep_supervision/`](deep_supervision/README.md) contains the complete two-stage
Attention-WNet project: image/mask loading, preprocessing and augmentation,
six auxiliary decoder heads, losses, training, checkpointing and evaluation.
Ten small synthetic image/mask pairs are included for an immediate CPU run.

Use Python 3.10 and run from that directory:

```bash
cd deep_supervision
python -m pip install -r requirements.txt
python train.py --device cpu
python evaluate.py --device cpu --save-predictions
python check_model.py --full-size
```

The default configuration runs two epochs with a small model. Outputs include
checkpoints, learning curves, test metrics and predicted masks. The synthetic
data demonstrates execution; it is not a medical benchmark. See the project
README and `config.research.json` for larger model settings and real-data setup.

## Architecture

The first Attention U-Net predicts vessel probabilities. These probabilities
are concatenated with the image channels and passed to a second Attention
U-Net for artery–vein predictions. Attention gates combine encoder features
with decoder features through skip connections. Each stage in the runnable
project has three additional decoder heads for deep supervision.

![Attention-WNet](https://github.com/user-attachments/assets/6557c2f6-d4b7-435e-8b1b-f86e6b95f45a)

The separate Swin-WNet implementation in the original notebook uses Swin
Transformer blocks and U-Net-style skip connections. It is not the model trained
by the `deep_supervision` scripts.

![Swin-WNet](https://github.com/user-attachments/assets/93bca168-5b48-4de8-b854-385c79187a0c)

## Other implementations

- [`pytorch/`](pytorch/) contains the earlier V11-compatible six-channel
  Attention-WNet inference package, preprocessing and sliding-window prediction,
  plus a separate Attention-U-Net with an auxiliary head. Its model and checkpoint
  contracts differ from the three-head cascade in `deep_supervision/`.
- [`attention-wnet-for-av-segmentation-of-fundus-image.ipynb`](attention-wnet-for-av-segmentation-of-fundus-image.ipynb)
  is the original TensorFlow/Keras research notebook. It is preserved separately
  and is not required for the runnable PyTorch example.
- [`Dataset/`](Dataset/) retains the existing retinal image and mask files.
  The CPU example uses only its own synthetic data.

The runnable project has 14 local checks covering model outputs, auxiliary-head
gradients, cascade gradients, the weighted objective, boundary gradients, spatial
alignment and data handling. The earlier PyTorch modules have 11 contract checks.
These checks establish software behavior, not reproduction of historical scores.

## Retinal data sources

- [DRIVE](https://drive.grand-challenge.org/)
- [HRF](https://www5.cs.fau.de/research/data/fundus-images/)

Follow each dataset's access and usage terms. Train, validation and test sets
must be separated appropriately before reporting research performance.
