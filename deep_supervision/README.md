# Attention-WNet with three-output deep supervision

This project trains a two-stage Attention-WNet for vessel and artery–vein
segmentation. Each stage has a main output and three auxiliary decoder outputs.
The first stage's vessel probabilities are concatenated with the image channels
before the second stage. All W-Net outputs are softmax probabilities.

## Run the complete example

Use Python 3.10. From this directory:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python train.py --device cpu
python evaluate.py --device cpu --save-predictions
python check_model.py --full-size
```

The included example contains **five real DRIVE image/mask pairs and five real
HRF image/mask pairs**, reused from the existing project dataset. There are six
training pairs (three from each dataset), two validation pairs (one each), and
two held-out test pairs (one each). Validation examples are reserved from the
source training folders; test examples come from the source test folders. No
selected image is repeated across the demonstration splits.

Images retain their aspect ratios and are resized to a maximum dimension of
512 pixels. Masks are resized to the same dimensions using nearest-neighbour
interpolation. The original data files are unchanged. For a short CPU run,
`config.json` uses 256×256 random training crops and centre evaluation crops,
small filter widths `[8, 16, 32, 64, 128]`, two epochs, four input channels
(CLAHE green plus LAB), and batch size two. The six auxiliary heads and cascade
structure are unchanged. This ten-image subset demonstrates the complete
pipeline; it is not an official dataset split or a reliable performance benchmark.

| Dataset | Training source files | Validation source file | Test source file |
| --- | --- | --- | --- |
| DRIVE_AV | `train/images/DRIVE0.png`, `DRIVE1.png`, `DRIVE2.png` | `train/images/DRIVE3.png` | `test/images/DRIVE0.png` |
| HRF_AV | `train/images/0.png`, `1.png`, `2.png` | `train/images/3.png` | `test/images/0.png` |

The names above are the existing project's filenames, not verified original
dataset identifiers. Each photograph is paired with the matching file in that
source folder's `label` directory. The original DRIVE release provides vessel
annotations; the A/V masks here are the additional annotations used in this
project. The synthetic generator is optional and writes to `synthetic_data/`,
so it does not overwrite the real samples.

### Dataset credits

- DRIVE images: Staal J, Abramoff MD, Niemeijer M, Viergever MA, van Ginneken B.
  *Ridge-based vessel segmentation in color images of the retina.* IEEE TMI,
  2004, 23(4):501–509. [Dataset](https://drive.grand-challenge.org/).
- HRF images: Budai A, Bock R, Maier A, Hornegger J, Michelson G.
  *Robust Vessel Segmentation in Fundus Images.* International Journal of
  Biomedical Imaging, 2013.
  [Dataset and CC BY 4.0 terms](https://www5.cs.fau.de/research/data/fundus-images/).
- Additional A/V annotation resources:
  [RITE](https://eye.medicine.uiowa.edu/rite-dataset) and
  [Hemelings et al., 2019](https://rubenhx.github.io/av-segmentation/).
  The bundled masks are taken from this project's existing `DRIVE_AV` and
  `HRF_AV` collections; the resources are listed separately from the image sources.

Dataset rights remain with their respective providers. These reduced examples
do not change the original dataset terms, and the code does not relicense them.

The optional `--full-size` check runs the default-width `[32, 64, 128, 256, 512]`
model at 608×608; it needs more memory than the example training run.

Outputs are written to `outputs/demo/`: `best.pth`, `last.pth`, `history.json`,
TensorBoard events, test metrics and optional predicted masks. No pretrained
checkpoint is bundled. Run `tensorboard --logdir outputs/demo/tensorboard` to
view the learning curves.

## Use retinal data

Edit `config.research.json` and set the three `dataset_splits` entries to your
data locations. Relative paths are resolved against the configuration file.
Each split has an `images` directory and a `labels` directory. Image and label
filenames must have matching stems; their extensions may differ. For example:

```text
retinal_data/
  train/images/001.png
  train/labels/001.png
  val/images/101.png
  val/labels/101.png
  test/images/201.png
  test/labels/201.png
```

Images are RGB. Masks use black for background, pure red for arteries and pure
blue for veins. Other nonblack mask colors count as vessels for stage one but
are excluded from artery–vein supervision and scoring. This permits unknown
vessels and crossings without assigning them an artery or vein class. Convert
any other annotation convention explicitly before using it.

Keep subjects, eyes and derived images separate across splits. The loader checks
that an image path is not reused in two splits; it cannot establish patient-level
independence. Training uses `val` for checkpoint selection. Evaluation defaults
to the separate `test` split.

```bash
python train.py --config config.research.json --device cuda
python evaluate.py --config config.research.json --device cuda --save-predictions
```

The research configuration uses 608×608 inputs and the original larger filter
widths. Set `hardware_profile`, image size, batch size, worker count and the
hyperparameter `gradient_accumulation_steps` for your hardware. Image size must
be at least 32 and divisible by 16. Supported input strategies are `resize`,
`dynamic_resize` (square resize or bottom/right padding), and `patching` (random
training crop and center evaluation crop). Evaluation reports scores on these
preprocessed images, not restored full-resolution masks.

## Losses and evaluation

The main and auxiliary outputs use weights `[1.0, 0.5, 0.25, 0.125]`. The stage
losses combine focal and Dice terms with optional differentiable Sobel-boundary
and total-variation terms. Stage two is supervised on labelled artery/vein
pixels; ground-truth vessel masks are used for this training loss only. Set
`hyperparameters.deep_supervision.enabled` to `false` to train just the main heads.
Partial final gradient-accumulation groups are stepped with their actual group
size. Elastic and grid transforms share the same spatial mapping across the
image, mask and RGB source used to calculate LAB channels.

At inference, stage one's **predicted** vessels gate the second stage's A/V
predictions. The evaluator reports end-to-end vessel/artery/vein scores and a
separately named conditional A/V score on ground-truth vessel pixels. Conditional
scores do not measure autonomous vessel detection. Counts are pooled across
the evaluated images; undefined ratios are recorded as JSON `null`.

The runnable project includes corrections to spatial augmentation, boundary
gradients, checkpoint/model matching and train/validation/test handling. Its
results therefore should not be presented as a reproduction of historical
experiment scores. Scores on this small demonstration subset verify execution only.

## Files

| File | Purpose |
| --- | --- |
| `models.py` | Cascaded Attention-WNet and six auxiliary heads |
| `dataset.py` | Paired loading, preprocessing, labels and augmentation |
| `losses.py` | Focal, Dice, Sobel boundary and total-variation losses |
| `objective.py` | Weighted losses across both stages and auxiliary heads |
| `train.py` | Training, validation, TensorBoard and checkpoints |
| `evaluate.py` | Held-out evaluation and predicted-mask export |
| `runtime.py` | Model construction, device/seed setup and output paths |
| `check_model.py` | Model, loss, preprocessing and split checks |
| `config.json` | Small CPU example |
| `config.research.json` | Larger model with editable retinal-data paths |
| `generate_sample_data.py` | Optional synthetic image/mask generator in a separate directory |
