# Evaluations

For evaluations of our models, we use modules provided by the WebNLG 2020 Challenge:
* Graph-to-Text (G2T) Evaluation: <https://github.com/WebNLG/GenerationEval>
* Text-to-Graph (T2G) Evaluation: <https://github.com/WebNLG/WebNLG-Text-to-triples>

## Install for Graph-to-Text (G2T)

Graph-to-Text task (aka RDF2Text for WebNLG) can be scored by installing this official WebNLG module:

```
cd ./eval
git clone https://github.com/WebNLG/GenerationEval.git
```

The code unfortunately will not work *as-is*.  
You need to apply the following edits from patch file [eval.py.patch](./eval.py.patch)
generated from git diff. Make sure the commit short SHA for the repos HEAD is `8067849` before applying the patch. 
To apply the patch, just do
```
# cwd is ./eval
cd ./GenerationEval
patch eval.py ../eval.py.patch
```

You now *must* install the dependencies for `GenerationEval` to work properly:
 ```
# cwd is ./eval
cd ./GenerationEval
bash ./install_dependencies.sh
```

## Install for Text-to-Graph (T2G)

Text-to-Graph task (aka Text2RDF for WebNLG) can be scored by installing this official WebNLG module:

```
cd ./eval
git clone https://github.com/WebNLG/WebNLG-Text-to-triples.git
```

For RL training for T2G, we need to use some official metrics as reward.  
We provide a patch to expose these metrics to the training code.  
Make sure that the commit short SHA for the repos HEAD is `6a40950` before applying the patch.
```
# cwd is ./eval
cd ./WebNLG-Text-to-triples
patch Evaluation_script.py ../Evaluation_script.py.patch
```

# Evaluation Scripts


## G2T Evaluation

For test/val splits, you can evaluate/score the generated text output by using our scripts:

```
bash ./scripts/evaluate.sh -j 1 -s testB -d webnlg -t ce  # where 1 is the epoch/jobid number
bash ./scripts/evaluate.sh -j 1 -s valB  -d webnlg -t ce  # same for valB
```
This script is a wrapper equivalent of running the following commands by hand:

For `testB`,
```
cd ./eval/GenerationEval
python ./eval.py \
       --reference ../../corpora/webnlg-references/release_v3.0/en/references.test/reference \
       --hypothesis ../.././output_eval.webnlg.testB/01/generate/hypothesis \
       -m bleu,meteor,chrf++ \
       -nr 4
```
and for `valB`
```
cd ./eval/GenerationEval
python ./eval.py \
       --reference ../../corpora/webnlg-references/release_v3.0/en/references.val/reference \
       --hypothesis ../../output_eval.webnlg.valB.ce/01/generate/hypothesis \
       -m bleu,meteor,chrf++ \
       -nr 4
```


## T2G Evaluation

For test/val splits, you can evaluate/score the generated graph output by using:

```
bash ./scripts/evaluate.sh -j 1 -s testA -d webnlg -t ce  # where 1 is the epoch/jobid number
bash ./scripts/evaluate.sh -j 1 -s valA  -d webnlg -t ce
```

This is equivalent to run by hand, for `testA`

```
cd ./eval/WebNLG-Text-to-triples 
python ./Evaluation_script.py \
       ../../corpora/webnlg-references/release_v3.0/en/references.rdf.test/test.xml \
       ../../output_eval.webnlg.testA.ce/01/generate/hyps.xml
```
and for `valA`

```
cd ./eval/WebNLG-Text-to-triples 
python ./Evaluation_script.py \
       ../../corpora/webnlg-references/release_v3.0/en/references.rdf.val/val.xml \
       ../../output_eval.webnlg.valA.ce/01/generate/hyps.xml
```

Note: Graph evaluations can have *very* long computation time.

