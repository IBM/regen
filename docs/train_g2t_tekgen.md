# G2T Training for TekGen

## Overview

We start from a `large` T5 Pretrained Language Model (PLM) from [HuggingFace](https://huggingface.co/t5-large).
1. CE fine-tuning of PLM. This is our "CE" model
1. SCST fine-tuning of CE model. This is our "SCST" model 

Training code has been written using [PyTorch](https://pytorch.org/) and can handle multiple GPUs training out-of-the-box if you have multiple GPUs on the same machine.  
We give instructions for training and evaluating CE and SCST models on the Tekgen dataset in sections below.

After CE and SCST fine-tunings, you should reach the ballpark of the performances reported below:

| model | epoch | BLEU | BLEU NLTK | METEOR | chrf++ |
| --:   | :---  | :--- | :---      | :---   | :---   |
| CE    | 5     | 19.92 | 0.20 | 0.21 | 0.37 |
| SCST  | 6.660 | 21.9  | 0.22 | 0.22 | 0.38 |

Setup: 1 machine using 4 cores and 4 A100 GPUs  
Note that results will depend on your hardware configuration (how many GPUs).  
There *will* be variations on the final results performance.

# CE Training

First, we do multiple epochs of CE fine-tuning starting from a `large` T5 model kindly provided by HuggingFace

## CE Fine-tuning

For CE fine-tuning of our `large` T5 PLM, please use:

```
bash ./scripts/train.tekgen.g2t.ce.sh -j  1  # epoch 1
bash ./scripts/train.tekgen.g2t.ce.sh -j  2
bash ./scripts/train.tekgen.g2t.ce.sh -j  3
bash ./scripts/train.tekgen.g2t.ce.sh -j  4
bash ./scripts/train.tekgen.g2t.ce.sh -j  5
```
This will run a training job for each of the first 5 epochs (early stopping).  
The script checkpoints (saves) the state of training at the end of each epoch and will restart from it on the next epoch.

We select a model by early stoping after the 5th epoch of PLM fine-tuning.  
Outputs from the training are organized into the following tree:

```
./output.tekgen.g2t.ce
├── checkpoints       # checkpoints of models
├── checkpoints_eval  # checkpoints for mid-epoch evaluations
├── jsonlog           # logs in .json format from rank 0 (machine readable)
│   ├── 01
│   ├── 02
│   ├── ...
│   └── 05
├── logs              # logs from logging modul for all GPUs
│   ├── 01
│   ├── 02
│   ├── ...
│   └── 05
└── tb                # tensorboard events file from rank 0
    ├── 01
    ├── 02
    ├── ...
    └── 05
```

For each job (or epoch), a model, state, and args file are saved in `./output.tekgen.g2t.ce/checkpoints`. 
At the 5th epoch, the last model checkpointed, you will have:
```
# cwd is ./output.tekgen.g2t.ce/checkpoints
args.epoch.5.0.iter.15625.pkl   # training arguments 
model.epoch.5.0.iter.15625.pth  # model weights
state.5.pkl                     # model training state
```

Note: The configuration for this CE training is defined in:
```
./cfg/train.tekgen.g2t.ce.cfg
```

## Evaluation of CE models

To evaluate a CE fine-tuned model, we need to:
1. generate hypotheses for our test/val set
1. score hypotheses using the *official* WebNLG evaluation code

Make sure you have installed the evaluation code [Evaluation](../eval/README.md)

### Generation for CE models

To generate hypotheses for each model from the 5 epochs of training, we use:  
``` 
for i in $(seq 1 5)
do 
    bash ./scripts/generate.sh -j $i -s testB -d tekgen -t ce
done
```
Note:
* You *must* have a GPU for running generation
* Generation can be slow... but each script can be run in parallel

The script `./scripts/generate.sh` can handle multiple splits (test/val), datasets (webnlg/tekgen) and training type (CE/SCST)  
For more information about all these options, do:
```
  bash ./scripts/generate.sh -h
```

Once generation is done, hypotheses can be found for each jobid/epoch in:
```
./output_eval.tekgen.testB.ce/01/generate/hypothesis
...
./output_eval.tekgen.testB.ce/05/generate/hypothesis
 ```

### Scoring for CE models

To score the generated hypotheses from the previous step, we use:
```
for i in $(seq 1 5)
do 
    bash ./scripts/evaluate.sh -j $i -s testB -d tekgen -t ce
done
```
Note:
* No GPU is needed for scoring -- the official WebNLG code does not use GPU
* Scoring is relatively fast... but each script can be run in parallel


All scoring outputs will be found in:
```
./score_eval.tekgen.testB.ce/01/score.out
...
./score_eval.tekgen.testB.ce/05/score.out
```

To extract results from scoring output, use:
```
for f in $(find ./score_eval.tekgen.testB.ce -name 'score.out' | sort -n)
do
    echo $f $(cat $f |  grep -A1 -B1 -e '-------' | grep -v B | grep -v -e '--' 2>&1)
done
```

Training a `large`T5 model w/ CE fine-tuning out-of-the-box should give you this range of results:

| scoring file | BLEU | BLEU NLTK | METEOR | chrf++ |
| :---         | :--- | :---      | :---   | :---   |
| ./score_eval.tekgen.testB.ce/01/score.out | 17.63 | 0.18 | 0.20 | 0.35 | 
| ./score_eval.tekgen.testB.ce/02/score.out | 18.69 | 0.19 | 0.21 | 0.36 | 
| ./score_eval.tekgen.testB.ce/03/score.out | 19.31 | 0.19 | 0.21 | 0.36 | 
| ./score_eval.tekgen.testB.ce/04/score.out | 19.59 | 0.20 | 0.21 | 0.36 | 
| ./score_eval.tekgen.testB.ce/05/score.out | 19.92 | 0.20 | 0.21 | 0.37 | 

Note: These results are given as reference. Keep in mind that there will be variations 
when running you own training.


# SCST Training

SCST training needs to start from a good CE model (we need a good policy for RL)  
We will start from CE epoch 5 since we decided to do early stopping on the CE training.  
Note that you could also select the best CE model on the validation set `valB`

## SCST Fine-tuning

For SCST fine-tuning our CE model, a `large` T5 PLM, you should use:

```
bash ./scripts/prepare.tekgen.g2t.scst.sh      # prepare CE model for SCST tree
bash ./scripts/train.tekgen.g2t.scst.sh  -j 6  # epoch 6
bash ./scripts/train.tekgen.g2t.scst.sh  -j 7  # epoch 7
```
The first step prepares (i.e. copy or symlink) the starting CE model into the SCST training directory tree  
Note that the starting CE model is specified in the config `./cfg/train.tekgen.g2t.scst.cfg`
```
cat ./cfg/train.tekgen.g2t.scst.cfg | grep scst_checkpoint
--scst_checkpoint_id       5
--scst_checkpoint_dir      ./output.tekgen.g2t.ce/checkpoints/
```
Then, the following steps perform SCST training on epoch 6, then epoch 7.  
SCST training is computationally more demanding than CE training, but usually the best systems are found in the first few epochs  
We checkpoints models during these training steps more often and evaluate them as described in the next section.  
All checkpointed models are saved in `./output.tekgen.g2t.scst/checkpoints_eval` during training.

## Evaluation of SCST models

### Generation for SCST models

To generate hypotheses for all models checkpointed during SCST training, we use:  

``` 
for f in $(find ./output.tekgen.g2t.scst/checkpoints_eval -name 'state.*.pkl' | sort -n)
do
    bash ./scripts/generate.checkpoints_eval.sh -f $f -s testB -d tekgen -t scst
done
```
Note:
* 1 GPU is needed for running the generation
* Generation can be *very* slow... If you can, parallelize this loop
The script `./scripts/generate.checkpoints_eval.sh` generates hypotheses for models checkpointed in the middle of an epoch training.  
You just need to specify a state file, a dataset split, a dataset, and 'scst' for model_type.  
For more information about all these options, do:
```
  bash ./scripts/generate.checkpoints_eval.sh -h
```


Once generation is done, hypotheses can be found for each checkpointed fractional epochs in:
```
./output_eval_checkpoints.tekgen.testB.scst/5.0600000/generate/hyps.xml
./output_eval_checkpoints.tekgen.testB.scst/5.2200000/generate/hyps.xml
...
./output_eval_checkpoints.tekgen.testB.scst/6.9800000/generate/hyps.xml
 ```
Note: epochs are fractional epochs (floating point numbers since we are checkpointing models along the training for an epoch of data)


### Scoring for SCST models

To score the generated hypotheses from the previous step, we use:
```
for f in $(find ./output_eval_checkpoints.tekgen.testB.scst/ -name 'hypothesis' | sort -n)
do
    bash ./scripts/evaluate.checkpoint_eval.sh -f $f -s testB -d tekgen -t scst
done
```
Note:
* you don't need any GPU for scoring -- official WebNLG code does not use GPU
* Scoring is relatively fast... but you can still parallelize this step


All output scores will be found in:
```
./score_eval_checkpoints.tekgen.testB.scst/5.0600000/score.out
./score_eval_checkpoints.tekgen.testB.scst/5.2200000/score.out
./score_eval_checkpoints.tekgen.testB.scst/5.3800000/score.out
... 
./score_eval_checkpoints.tekgen.testB.scst/6.6600000/score.out
./score_eval_checkpoints.tekgen.testB.scst/6.8200000/score.out
./score_eval_checkpoints.tekgen.testB.scst/6.9800000/score.out
```

To extract results from scoring outputs, use:
```
for f in $(find ./score_eval_checkpoints.tekgen.testB.scst -name 'score.out' | sort -n);
do
    echo $f $(cat $f |  grep -A1 -B1 -e '-------' | grep -v B | grep -v -e '--' 2>&1)
done
```

SCST Training out-of-the-box should give you results close to these:

| scoring file | BLEU | BLEU NLTK | METEOR | chrf++ |
| :---         | :--- | :---      | :---   | :---   |
| ./score_eval_checkpoints.tekgen.testB.scst/5.0600000/score.out | 20.04 | 0.2  | 0.21 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/5.2200000/score.out | 20.41 | 0.2  | 0.21 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/5.3800000/score.out | 20.17 | 0.2  | 0.21 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/5.5400000/score.out | 20.44 | 0.2  | 0.21 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/5.7000000/score.out | 20.46 | 0.21 | 0.22 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/5.8600000/score.out | 20.21 | 0.2  | 0.21 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/6.0200000/score.out | 19.96 | 0.2  | 0.21 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/6.1800000/score.out | 19.94 | 0.2  | 0.21 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/6.3400000/score.out | 20.74 | 0.21 | 0.21 | 0.37 | 
| ./score_eval_checkpoints.tekgen.testB.scst/6.5000000/score.out | 21.36 | 0.21 | 0.22 | 0.38 | 
| ./score_eval_checkpoints.tekgen.testB.scst/6.6600000/score.out | 21.9  | 0.22 | 0.22 | 0.38 | 
| ./score_eval_checkpoints.tekgen.testB.scst/6.8200000/score.out | 21.73 | 0.22 | 0.22 | 0.38 | 
| ./score_eval_checkpoints.tekgen.testB.scst/6.9800000/score.out | 21.54 | 0.22 | 0.22 | 0.38 | 

Note: These results are given to you as reference as they will vary depending on your setup (number of GPUs, etc.)  
We only show these results to check your training setup.--
It is important in a real experimental setup to select your best SCST model based on your results on the validation set `valB` and *not directly* on `testB`.  
For `valB`, all the steps for generation and scoring are the same except you should use `valB` instead of `testB` for dataset split.

