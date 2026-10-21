# m6A-MAC

m6A-MAC predicts the probability of m6A at the central adenosine of a fixed-length
RNA sequence. A frozen RiNALMo-Mega encoder supplies residue representations to a
trainable MAC U-Net combining attention, Mamba and convolution.

## Installation

Use Python 3.10 or later and a PyTorch installation compatible with your CUDA
runtime. Install this package from the repository root:

```bash
python -m pip install -e .
```

Install the official [RiNALMo package](https://github.com/lbcb-sci/RiNALMo)
separately, following its installation instructions. Install `mamba-ssm` and
`causal-conv1d` builds compatible with your PyTorch and CUDA versions. The model
uses the non-flash RiNALMo attention implementation with PyTorch scaled dot
product attention. A native Mamba installation is required for the MAC head.

Obtain the official RiNALMo-Mega pretrained checkpoint and place it at a location
you choose. The example configuration uses `weights/rinalmo_mega_pretrained.pt`.
Data and weights are supplied separately; the code does not download them.

## Repository structure

```text
configs/train.json          Training configuration example
run.py                      Training and inference commands
src/m6a/models/             MAC U-Net, frozen encoder and input validation
src/m6a/training/train.py   Training, validation and checkpoint selection
tests/                      Model, data and portability tests
licenses/                   Third party attribution
```

All example file locations are relative to the current working directory.
Commands below run directly from this repository root, including before an
editable installation. No parent research directory or external symlink is used.

## Data

Prepare `data/train.tsv` and `data/validation.tsv`. Each tab-separated file needs
`id`, `label` and `seq201` columns for a 201 nucleotide run. For other supported
lengths, use `seq101`, `seq401` or `seq801`. Labels are 0 or 1, each split must
contain both classes, IDs must be unique within a split, and sequences must have
exactly the selected length with A at the central position. Lowercase and U are
normalized to uppercase and T. Ambiguous bases and implicit padding are rejected.
If `seq401` is absent, it can be derived from the exact center of `seq801` before
encoding.

Training and validation must have disjoint IDs and exact input sequences. When
both files include a `gene` column, gene overlap is rejected as well. Supply
scientifically appropriate splits; these checks do not establish independence
from encoder pretraining data. No automatic split construction is performed.

## Training

Edit `configs/train.json` for your data, checkpoint, device and output directory.
First validate the configuration and input tables:

```bash
python run.py train --config configs/train.json --dry-run
```

The dry run checks data and checkpoint existence without loading the encoder.
It does not test checkpoint compatibility, CUDA kernels or training performance.
Start training with:

```bash
python run.py train --config configs/train.json
```

Each seed trains a fresh MAC head while the encoder remains frozen and in
evaluation mode. Representations are computed on demand in float32, cast through
float16, and supplied to the head in float32. This release does not use a persistent
embedding cache, so extraction is repeated during training.

The trainer uses AdamW, gradient accumulation to an effective batch of 256,
gradient clipping at 1, validation average precision for checkpoint selection,
and early stopping after five epochs without improvement. The learning rate is
halved after validation AP plateaus. The classification threshold maximizes
validation F1, choosing the largest threshold on ties. Test labels are not used.

Each seed directory contains `best.pt` with the downstream head, a protocol with
input and encoder SHA-256 hashes, training history, and predictions from the best
validation epoch. Existing seed directories are protected, including interrupted
runs. Choose a new output directory to restart; automatic resume is not supported.
This is a standalone trainer and does not claim numerical reproduction of a
particular previously registered experiment.

## Inference

Load the same encoder checkpoint, input length and head dimension used in
training. The command uses a synthetic central-A sequence when `--sequence` is
omitted:

```bash
python run.py predict --encoder-checkpoint weights/rinalmo_mega_pretrained.pt --checkpoint runs/mac201/seed17/best.pt --length 201 --dimension 128 --device cuda:0
```

Supply one or more exact-length sequences with `--sequence`. Omitting
`--checkpoint` uses a random head and reports that status in the output.

After installation, the Python API is:

```python
from m6a.models import M6AMAC

model = M6AMAC(length=201, dimension=128,
               encoder_checkpoint="weights/rinalmo_mega_pretrained.pt",
               device="cuda:0")
model.load_checkpoint("runs/mac201/seed17/best.pt")
model.eval()
probabilities = model(["C" * 100 + "A" + "G" * 100])
```

## Tests

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

Tests substitute the large pretrained encoder and native Mamba operations where
needed. They verify the frozen gradient boundary, training artifacts, input
validation, output protection and execution from a relocated repository. Actual
RiNALMo weights and native CUDA kernels require a separate integration run.

## Attribution

RiNALMo is an external dependency from the
[official RiNALMo project](https://github.com/lbcb-sci/RiNALMo); its Apache 2.0
license is retained in `licenses/RiNALMo.LICENSE`. See `licenses/NOTICE.md` for
implementation attribution. This repository does not assign a new license to
third party work.
