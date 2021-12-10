# T2G Training for WebNLG

## Overview

We start from a `large` T5 Pretrained Language Model (PLM) from [HuggingFace](https://huggingface.co/t5-large).
1. CE fine-tuning of PLM. This is our "CE" model
1. SCST fine-tuning of CE model. This is our "SCST" model 

Training code has been written using [PyTorch](https://pytorch.org/) and can handle multiple GPUs training out-of-the-box if you have multiple GPUs on the same machine.  
We give instructions for training and evaluating CE and SCST models on the WebNLG dataset in sections below.

After CE and SCST fine-tunings, you should reach the ballpark of the performances reported below:


| model | epoch |   F1 | Precision | Recall |
| --:   | :---  | :--- | :---      | :---   |
| CE    | 5     | 0.7011 | 0.6920 | 0.7163 |
| SCST  | 5.095 | 0.7091 | 0.7000 | 0.7244 |

Setup: 1 machine using 4 cores and 4 A100 GPUs  
Note that results will depend on your hardware configuration (how many GPUs).  
There *will* be variations on the final results performance.

# CE Training

First, we do multiple epochs of CE fine-tuning starting from a `large` T5 model kindly provided by HuggingFace

## CE Fine-tuning

For CE fine-tuning of our `large` T5 PLM, please use:

```
bash ./scripts/train.webnlg.t2g.ce.sh -j  1  # epoch 1
bash ./scripts/train.webnlg.t2g.ce.sh -j  2
bash ./scripts/train.webnlg.t2g.ce.sh -j  3
bash ./scripts/train.webnlg.t2g.ce.sh -j  4
bash ./scripts/train.webnlg.t2g.ce.sh -j  5
```
This will run a training job for each of the first 5 epochs (early stopping).  
The script checkpoints (saves) the state of training at the end of each epoch and will restart from it on the next epoch.

We select a model by early stoping after the 5th epoch of PLM fine-tuning.  
Outputs from the training are organized into the following tree:

```
./output.webnlg.t2g.ce
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

For each job (or epoch), a model, state, and args file are saved in `./output.webnlg.t2g.ce/checkpoints`. 
At the 5th epoch, the last model checkpointed, you will have:
```
# cwd is ./output.webnlg.t2g.ce/checkpoints
args.epoch.5.0.iter.40460.pkl   # training arguments 
model.epoch.5.0.iter.40460.pth  # model weights
state.5.pkl                     # model training state
```

Note: The configuration for this CE training is defined in:
```
./cfg/train.webnlg.t2g.ce.cfg
```

## Evaluation of CE models

To evaluate a CE fine-tuned model, we need to:
1. generate hypotheses for our test/val set
1. score hypotheses using the *official* WebNLG evaluation code

Make sure you have installed the evaluation code [Evaluation](../eval/README.md), and that WebNLG references are available [Corpora](../corpora/README.md)

### Generation for CE models

To generate hypotheses for each model from the 5 epochs of training, we use:  
``` 
for i in $(seq 1 5)
do 
    bash ./scripts/generate.sh -j $i -s testA -d webnlg -t ce
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
./output_eval.webnlg.testA.ce/01/generate/hypothesis
...
./output_eval.webnlg.testA.ce/05/generate/hypothesis
 ```

### Scoring for CE models

To score the generated hypotheses from the previous step, we use:
```
for i in $(seq 1 5)
do 
    bash ./scripts/evaluate.sh -j $i -s testA -d webnlg -t ce
done
```
Note:
* No GPU is needed for scoring -- the official WebNLG code does not use GPU
* Scoring is relatively fast... but each script can be run in parallel


All scoring outputs will be found in:
```
./score_eval.webnlg.testA.ce/01/score.out
...
./score_eval.webnlg.testA.ce/05/score.out
```

To extract the results from scoring outputs, use:
```
for f in $(find ./score_eval.webnlg.testA.ce/ -name 'score.out' | sort -n)
do
    f1m=$(cat $f | grep -A25 -e'Total scores'  | grep -A5 -e'Exact' | grep F1)
    prm=$(cat $f | grep -A25 -e'Total scores'  | grep -A5 -e'Exact' | grep Precision)
    echo $f $f1m $prm
done
```



Training a `large`T5 model w/ CE fine-tuning out-of-the-box should give you this range of results:

| scoring file |    F1 | Precision | Recall |
| :---         | :---  | :---      | :---   |
| ./score_eval.webnlg.testA.ce/01/score.out | 0.6518 | 0.6435 | 0.6662 |
| ./score_eval.webnlg.testA.ce/02/score.out | 0.6726 | 0.6633 | 0.6881 |
| ./score_eval.webnlg.testA.ce/03/score.out | 0.6906 | 0.6815 | 0.7060 |
| ./score_eval.webnlg.testA.ce/04/score.out | 0.6916 | 0.6822 | 0.7077 |
| ./score_eval.webnlg.testA.ce/05/score.out | 0.7011 | 0.6920 | 0.7163 |

Note: These results are given to you as reference as they will vary depending on your setup (number of GPUs, etc.)

# SCST Training

SCST training needs to start from a good CE model (we need a good policy for RL)  
We will start from CE epoch 5 since we decided to do early stopping on the CE training.  
Note that you could also select the best CE model on the validation set `valA`

## SCST Fine-tuning

For SCST fine-tuning our CE model, a `large` T5 PLM, you should use:

```
bash ./scripts/prepare.webnlg.t2g.scst.sh      # prepare CE model for SCST tree
bash ./scripts/train.webnlg.t2g.scst.sh  -j 6  # epoch 6
```
The first step prepares (i.e. copy or symlink) the starting CE model into the SCST training directory tree  
Note that the starting CE model is specified in the config `./cfg/train.webnlg.t2g.scst.cfg`
```
cat ./cfg/train.webnlg.t2g.scst.cfg | grep scst_checkpoint
--scst_checkpoint_id       5
--scst_checkpoint_dir      ./output.webnlg.t2g.ce/checkpoints/
```
Then, the following step performs SCST training on epoch 6  
SCST training is computationally more demanding than CE training, but usually the best systems are found in the first few epochs  
We checkpoints models during these training steps more often and evaluate them as described in the next section.  
All checkpointed models are saved in `./output.webnlg.t2g.scst/checkpoints_eval` during training.

## Evaluation of SCST models

### Generation for SCST models

To generate hypotheses for all models checkpointed during SCST training, we use:  

``` 
for f in $(find ./output.webnlg.t2g.scst/checkpoints_eval -name 'state.*.pkl' | sort -n)
do
    bash ./scripts/generate.checkpoints_eval.sh -f $f -s testA -d webnlg -t scst
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
./output_eval_checkpoints.webnlg.testA.scst/5.1942186/generate/hypothesis
...
./output_eval_checkpoints.webnlg.testA.scst/7.2267389/generate/hypothesis
 ```
Note: epochs are fractional epochs (floating point numbers since we are checkpointing models along the training for an epoch of data)


### Scoring for SCST models

To score the generated hypotheses from the previous step, we use:
```
for f in $(find ./output_eval_checkpoints.webnlg.testA.scst/ -name 'hypothesis' | sort -n)
do
    bash ./scripts/evaluate.checkpoint_eval.sh -f $f -s testA -d webnlg -t scst
done
```
Note:
* you don't need any GPU for scoring -- official WebNLG code does not use GPU
* Scoring is relatively fast... but you can still parallelize this step


All output scores will be found in:
```
./score_eval_checkpoints.webnlg.testA.scst/5.0333642/score.out
./score_eval_checkpoints.webnlg.testA.scst/5.0951498/score.out
./score_eval_checkpoints.webnlg.testA.scst/5.1569354/score.out
./score_eval_checkpoints.webnlg.testA.scst/5.2187210/score.out
./score_eval_checkpoints.webnlg.testA.scst/5.2805066/score.out
./score_eval_checkpoints.webnlg.testA.scst/5.3422922/score.out
...
./score_eval_checkpoints.webnlg.testA.scst/5.9601483/score.out 
```

To extract the results from scoring outputs, use:
```
for f in $(find ./score_eval_checkpoints.webnlg.testA.scst/ -name 'score.out' | sort -n)
do
    f1m=$(cat $f | grep -A25 -e'Total scores'  | grep -A5 -e'Exact' | grep F1)
    prm=$(cat $f | grep -A25 -e'Total scores'  | grep -A5 -e'Exact' | grep Precision)
    echo $f $f1m $prm
done
```

SCST Training out-of-the-box should give you results close to these:

| scoring file |    F1 | Precision | Recall |
| :---         | :---  | :---      | :---   |
| ./score_eval_checkpoints.webnlg.testA.scst/5.0333642/score.out | 0.7035 | 0.6945 | 0.7185 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.0951498/score.out | 0.7091 | 0.7000 | 0.7244 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.1569354/score.out | 0.6987 | 0.6896 | 0.7140 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.2187210/score.out | 0.6878 | 0.6788 | 0.7027 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.2805066/score.out | 0.6947 | 0.6856 | 0.7098 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.3422922/score.out | 0.7044 | 0.6953 | 0.7198 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.4040778/score.out | 0.6031 | 0.5950 | 0.6169 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.4658635/score.out | 0.6789 | 0.6699 | 0.6942 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.5276491/score.out | 0.6992 | 0.6902 | 0.7146 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.5894347/score.out | 0.6753 | 0.6665 | 0.6902 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.6512203/score.out | 0.7017 | 0.6927 | 0.7167 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.7130059/score.out | 0.6884 | 0.6794 | 0.7038 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.7747915/score.out | 0.6870 | 0.6780 | 0.7021 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.8365771/score.out | 0.6926 | 0.6835 | 0.7080 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.8983627/score.out | 0.6847 | 0.6754 | 0.7005 |
| ./score_eval_checkpoints.webnlg.testA.scst/5.9601483/score.out | 0.6599 | 0.6509 | 0.6751 |

Note: These results are given to you as reference as they will vary depending on your setup (number of GPUs, etc.)  
It is important in a real experimental setup to select your best SCST model based on your results on the validation set `valA` and *not directly* on `testA`.  
For `valA`, all the steps for generation and scoring are the same except you should use `valA` instead of `testA` for dataset split.

