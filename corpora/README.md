# Datasets

## Quickstart

Install WebNLG and TekGen datasets
```
# webnlg
cd ./corpora
git clone https://gitlab.com/shimorina/webnlg-dataset.git
```
```
# tekgen
cd ./corpora
mkdir -p ./tekgen/official
cd ./tekgen/official
wget  https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-train.tsv
wget  https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-validation.tsv
wget  https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-test.tsv
```
Prepare WebNLG reference. 
```
# Prepare webnlg evaluations references

# note: make sure you in the git repos root
# text references
bash ./scripts/generate_references_test.sh
bash ./scripts/generate_references_val.sh
# graphs references
bash ./scripts/generate_rdf_references_test.sh
bash ./scripts/generate_rdf_references_val.sh
```

Detailed information are given sections below.



## WebNLG

We use the [WebNLG 2020 v3.0 Dataset (rev 3.0)](https://webnlg-challenge.loria.fr/challenge_2020/#data), 
distributed from the [gitlab repos](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0).

Get the official data .xml files:
```
cd ./corpora
git clone https://gitlab.com/shimorina/webnlg-dataset.git
```
Dataset for rev 3.0 is located in `./webnlg-dataset/release_v3.0`

## TekGen

The TekGen dataset is part of the [KELM Corpus](https://github.com/google-research-datasets/KELM-corpus) from Google Research and can be found here: <https://github.com/google-research-datasets/KELM-corpus#part-1-tekgen-training-corpus>

```
cd ./corpora
mkdir -p ./tekgen/official
cd ./tekgen/official
wget  https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-train.tsv
wget  https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-validation.tsv
wget  https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-test.tsv
```

# Evaluation References

## WebNLG References

Our experiments use WebNLG 2020 *rev 3.0*.  
Note: Papers often report results on "WebNLG", but from different revision numbers.  
Revision *rev 3.0* was used in the [WebNLG 2020 Challenge](https://webnlg-challenge.loria.fr/challenge_2020)


### Text References

For graph-to-text (G2T) generation evaluation, we need text references.

We provide scripts to generate references for `test` and `val` splits of WebNLG 2020 rev 3.0.  
Note: these scripts are slightly modified versions of the official WebNLG code from <https://gitlab.com/webnlg/corpus-reader.git>

```
bash ./scripts/generate_references_test.sh
bash ./scripts/generate_references_val.sh
```
The output logs are saved in `./tools/logs/generate_references.py.log`

References files are saved in `./corpora/webnlg-references/release_v3.0/en/`

```> tree ./corpora/webnlg-references/release_v3.0/en/
./corpora/webnlg-references/en
├── references.test
│   ├── reference0
│   ├── reference1
│   ├── reference2
│   ├── reference3
│   └── reference4
└── references.val
    ├── reference0
    ├── reference1
    ├── reference2
    ├── reference3
    ├── reference4
    ├── reference5
    ├── reference6
    └── reference7
```

### Graph References (RDF)

For text-to-graph (T2G) generation evaluation, we need graph/RDF references.

We provide scripts to generate references for `test` and `val` splits of WebNLG 2020 v3.0.  
Note: these scripts are slightly modified versions of official WebNLG code from <https://gitlab.com/webnlg/corpus-reader.git>

```
bash ./scripts/generate_rdf_references_test.sh
bash ./scripts/generate_rdf_references_val.sh
```
The output logs are saved in `./tools/logs/generate_references_rdf.py.log`

References files are saved in `./corpora/webnlg-references/release_v3.0/en/`

```> tree ./corpora/webnlg-references/release_v3.0/en/
./corpora/webnlg-references/release_v3.0/en/
├── references.rdf.test
│   └── test.xml
└── references.rdf.val
    └── val.xml
```

## TekGen References

For TekGen references, we use references provided from `quadruples-test.tsv`
