# ReGen: Reinforcement Generation for bi-directional Text and Knowledge Base using Pretrained Language Models

# Summary

This is the official code for ReGen from our [EMNLP21 paper](https://aclanthology.org/2021.emnlp-main.83/)  

For a quick introduction to ReGen, please read our [short description](./docs/regen.md) .
For a more detailed explanation, please check out our paper [ReGen: Reinforcement Learning for Text and Knowledge Base Generation using Pretrained Language Models](https://aclanthology.org/2021.emnlp-main.83/) or on [arXiv](https://arxiv.org/abs/https://arxiv.org/abs/2108.12472)

The code will be added to this repos on December 3rd, 2021


# Citation
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
