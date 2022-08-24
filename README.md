# Fine Tune

This repository contains code to fine tune a classifier using the aclImdb_v1 dataset. The resulting model can then be converted to ONNX (and used by Apache OpenNLP).

## Requirements

```
python3 -m pip install transformers onnxruntime torch sklearn
```

## Train

`python3 train.py`

## Convert to ONNX

`python3 -m transformers.onnx --model=local-pt-checkpoint/ --feature sequence-classification exported-to-onnx`
