# distilbert-imdb

This repository contains code to fine tune a classifier using the imdb dataset. The resulting model can then be converted to ONNX (and used by Apache OpenNLP).

For a trained model, see https://huggingface.co/jzonthemtn/distilbert-imdb.

## Requirements

```
python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install transformers onnxruntime torch sklearn
```

## Convert to ONNX

`python3 -m transformers.onnx --model=local-pt-checkpoint/ --feature sequence-classification exported-to-onnx`

## Training and Evaluation Summaries

| Epochs      | Training | Evaluation |
| ----------- | ---------- | ----------- |
| 1      |     | {'eval_loss': 0.21822933852672577, 'eval_accuracy': 0.93224, 'eval_f1': 0.9321042084168337, 'eval_runtime': 225.0407, 'eval_samples_per_second': 111.091, 'eval_steps_per_second': 13.886}      |
| 5 | {'train_runtime': 3537.2631, 'train_samples_per_second': 35.338, 'train_steps_per_second': 2.209, 'train_loss': 0.11836099642938479, 'epoch': 5.0}     | {'eval_loss': 0.3719911277294159, 'eval_accuracy': 0.93136, 'eval_f1': 0.9316987740805606, 'eval_runtime': 235.7789, 'eval_samples_per_second': 106.032, 'eval_steps_per_second': 6.629, 'epoch': 5.0} |

