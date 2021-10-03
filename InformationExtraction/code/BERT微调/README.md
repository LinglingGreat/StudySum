## Model Architecture

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)
- **If you want to use CRF layer, give `--use_crf` option**

## Dependencies

- python>=3.6
- torch==1.6.0
- transformers==3.0.2
- seqeval==0.0.12
- pytorch-crf==0.7.2

## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Training & Evaluation

```bash
$ python3 main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval \
                  --use_crf

# For ATIS
$ python3 main.py --task atis \
                  --model_type bert \
                  --model_dir atis_model \
                  --do_train --do_eval
# For Snips
$ python3 main.py --task snips \
                  --model_type bert \
                  --model_dir snips_model \
                  --do_train --do_eval

# python JointBERT/main.py --task atis --model_type bert --model_dir experiments/jointbert_0 --do_train --do_eval --train_batch_size 2
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## Results

- Run 5 ~ 10 epochs (Record the best result)
- Only test with `uncased` model
- ALBERT xxlarge sometimes can't converge well for slot prediction.

|           |                  | Intent acc (%) | Slot F1 (%) | Sentence acc (%) |
| --------- | ---------------- | -------------- | ----------- | ---------------- |
| **Snips** | BERT             | **99.14**      | 96.90       | 93.00            |
|           | BERT + CRF       | 98.57          | **97.24**   | **93.57**        |
|           | ALBERT + CRF     | 99.00          | 96.55       | 92.57            |
| **ATIS**  | BERT             | 97.87          | 95.59       | 88.24            |
|           | BERT + CRF       | **97.98**      | 95.93       | 88.58            |


