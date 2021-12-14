# ReGen: Reinforcement Generation for bi-directional Text and Knowledge Base using Pretrained Language Models

# Summary

This is the official code for ReGen from our [EMNLP21 paper](https://aclanthology.org/2021.emnlp-main.83/)  

For a quick introduction to ReGen, please read our [short description](./docs/regen.md).

For a more detailed explanation, please check our paper [ReGen: Reinforcement Learning for Text and Knowledge Base Generation using Pretrained Language Models](https://aclanthology.org/2021.emnlp-main.83/) or a more up to date version on [arXiv](https://arxiv.org/abs/https://arxiv.org/abs/2108.12472)

The code was released on December 10th, 2021


# ReGen

`ReGen` stands for *Reinforced Generation* for Graph-to-Text and Text-to-Graph generative tasks.  
Details are given in the long paper [ReGen: Reinforcement Learning for Text and Knowledge Base Generation using Pretrained Language Models](https://aclanthology.org/2021.emnlp-main.83/) that was presented as an oral presentation at EMNLP 2021 main conference. A version of the paper that will be updated w/ new results can be found on [arXiv](https://arxiv.org/abs/2108.12472).


Contacts:
- Pierre Dognin `pdognin{at_}us[dot_]ibm[_dot]com`
- Inkit Padhi `inkit.padhi[_at]ibm{dot}com`


---
## Requirements

* Python (3.8)
* PyTorch (1.7.1)
* transformers (4.2.2)
* nltk (>=3.6.3)

Install these packages in conda environment `regen` with:
```
conda env create -f setup.yml
```

Install nltk (>=3.6.3) by hand
```
conda activate regen
pip install --user 'nltk>=3.6.3'
```

Note: we assume in this documentation that you have activated the conda environment by using:
```
conda activate regen
```
---

# Datasets

Two datasets were used in our experimentss:
* [WebNLG 2020 Challenge (rev 3.0)](https://webnlg-challenge.loria.fr/challenge_2020)
* [TekGen](https://github.com/google-research-datasets/KELM-corpus) part of the KELM Corpus from Google Research.

To install the datasets, prepare references for evaluations, please look at the steps in [datasets documentation](./corpora/README.md)


# Evaluation Metrics

Evaluations of generation results are performed using the WebNLG offical evaluation packages for *both* datasets.  
For Reinforcement Learning training, rewards are derived from evaluation metrics and are required for RL fine-tuning models. 

Installation steps are given in [evaluation documentation](./eval/README.md)


# Training

We can train model for 2 generative tasks:
- G2T (graph-to-text) task that will generate text from a graph input
- T2G (text-to-graph) task that will generate a graph from input sentences

Models are trained from a Pretrained Language Model (PLM) T5 in two steps:
1. Fine-tuning of a T5 `large` model using Cross-Entropy (CE). This is our CE model
1. CE model is fine-tuned using Self-Critical Sequence Training (SCST) [(Rennie, et al., 2017)](#rennie2017), a variant of REINFORCE [(Williams, 1992)](#williams1992)

We give instructions on training a model on the WebNLG/TekgGen datasets below.  
The training scripts are based on [PyTorch](https://pytorch.org/) implementation using PLMs from [HuggingFace](https://huggingface.co)  
The code can handle multiple GPUs training out-of-the-box. 
Shell scripts to facilitate training and evaluation are provided [Scripts](./scripts/README.md)

## WebNLG Training

- G2T training: [G2T training documentation on WebNLG](./docs/train_g2t_webnlg.md)  
- T2G training: [T2G training documentation on WebNLG](./docs/train_t2g_webnlg.md)  

## TekGen Training

- G2T training: [G2T training documentation on TekGen](./docs/train_g2t_tekgen.md)  
- T2G training: (coming next week)

--- 

# How to Cite
```
@inproceedings{dognin-etal-2021-regen,
    title = "{R}e{G}en: {R}einforcement Learning for Text and Knowledge Base Generation using Pretrained Language Models",
    author = "Dognin, Pierre  and
      Padhi, Inkit  and
      Melnyk, Igor  and
      Das, Payel",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.83",
    pages = "1084--1099",
    abstract = "Automatic construction of relevant Knowledge Bases (KBs) from text, and generation of semantically meaningful text from KBs are both long-standing goals in Machine Learning. In this paper, we present ReGen, a bidirectional generation of text and graph leveraging Reinforcement Learning to improve performance. Graph linearization enables us to re-frame both tasks as a sequence to sequence generation problem regardless of the generative direction, which in turn allows the use of Reinforcement Learning for sequence training where the model itself is employed as its own critic leading to Self-Critical Sequence Training (SCST). We present an extensive investigation demonstrating that the use of RL via SCST benefits graph and text generation on WebNLG+ 2020 and TekGen datasets. Our system provides state-of-the-art results on WebNLG+ 2020 by significantly improving upon published results from the WebNLG 2020+ Challenge for both text-to-graph and graph-to-text generation tasks. More details at https://github.com/IBM/regen.",
}
```

## Bibliography

Rennie, S. J. et al., 2017. [*Self-critical Sequence Training for Image Captioning.*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.pdf)
Honolulu, Hawaii, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
<a name="rennie2017"></a>

Williams, R. J., 1992. [*Simple Statistical Gradient-following Algorithms for Connectionist Reinforcement Learning*](https://link.springer.com/article/10.1007/BF00992696).
*Machine learning,* 8(3), pp. 229-256.
<a name="Williams1992"></a>
