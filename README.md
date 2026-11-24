# Swedish grammaticality scoring(for AI Sweden thesis task)

A model that gives a Swedish text a score between 0 (ungrammatical) and 1 (grammatical).
It is [AI-Sweden-Models/ModernBERT-base](https://huggingface.co/AI-Sweden-Models/ModernBERT-base)
(150M parameters) fine-tuned as a regression model on Swedish learner and treebank data.

## Results

On the 120 sentences in `data/test.jsonl`, measured with `evaluate.py` on one NVIDIA A30 (24 GB):

| Model | Params | Spearman | ROC-AUC | Samples/min |
|---|---|---|---|---|
| Baseline (ModernBERT-large embeddings + SVR) | 395M | 0.528 | 0.783 | 562 |
| **This model (fine-tuned ModernBERT-base)** | **150M** | **0.728** | **0.921** | 295 |


## Links

- Model: 
- Dataset: 

## Setup

Tested with Python 3.12 on Linux, on the University of Gothenburg's GPU server (mltgpu-2, NVIDIA A30 24 GB).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Score texts (downloads the model from Hugging Face). Each input line is `{"id": ..., "text": "..."}`:

```bash
python predict.py --input data/test.jsonl --output scores.jsonl
```

Evaluate on the test set (runs `predict.py`, then prints Spearman, ROC-AUC and speed):

```bash
python evaluate.py
```

To rebuild everything from scratch:

```bash
python make_data.py   # builds data/my_train.jsonl (downloads DaLAJ and ScaLA)
python train.py       # fine-tunes the model, saves it to my_model/ (about 6 min on one A30)
```

## Data

15,606 examples from three sources:

| Source | Rows | What it is | License |
|---|---|---|---|
| training data (`data/train.jsonl`) | 478 | the task's own examples, with graded scores | from the task |
| [DaLAJ](https://huggingface.co/datasets/mteb/DalajClassification) | 7,682 | sentences by Swedish learners, correct or with one real error | CC BY 4.0 |
| [ScaLA-sv](https://huggingface.co/datasets/mteb/scala_sv_classification) | 7,446 | treebank sentences, some broken by swapping or deleting words | CC BY-SA 4.0 |

DaLAJ and ScaLA only have correct/incorrect labels. I turned them into scores using the toy data:
correct sentences there have a mean score of about 0.95, and sentences with one error about 0.39.
So correct = 0.95 and incorrect = 0.4.

## Training

- Regression with one output, MSE loss (the Hugging Face `Trainer` default for `num_labels=1`)
- 3 epochs, learning rate 2e-5 with 10% warmup, batch size 32, max length 128 tokens
- 90/10 train/validation split, final validation Spearman 0.51
- Predictions are clipped to [0, 1] in `predict.py`

## Reproducibility

- Seed 42 for data shuffling, the train/validation split and training (`set_seed`)
- Pinned Hugging Face revisions (commit hashes) for both datasets and the base model
- Pinned package versions in `requirements.txt`

## Known issues

- A message `fatal error: Python.h: No such file or directory` from Triton can appear. It is harmless, and
  training and prediction still work. (`reference_compile=False` turns off ModernBERT's torch.compile.)
- On a machine with several GPUs, training can hang. Run it on one GPU: `CUDA_VISIBLE_DEVICES=0 python train.py`