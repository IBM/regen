# ReGen: Reinforcement Learning for Text and Knowledge Base Generation using Pretrained Language Models

Pierre Dognin, Inkit Padhi, Igor Melnyk, Payel Das

# Introduction

One foundational goal of IBM Research is to process real-word data in
all its modalities: text, graph, tabular data, time series, etc. All are
important representations of knowledge commonly observed in
information-centric applications. Information should be easily
translated from one modality to another seamlessly, without compromising
on its factual content. For example, representing facts from a text into
a knowledge graph is a fundamental principle of knowledge
representation. In this work, we focus on accurate bi-directional
transfer of information between graph and text modalities.

* Text to Graph: Automatic Generation of Knowledge Bases (KBs)

![](https://github.com/IBM/regen/blob/assets/imgs/t2g.svg)

* Graph to Text: Text generation from KBs

![](https://github.com/IBM/regen/blob/assets/imgs/g2t.svg)

The text-to-graph (T2G) information transfer is a crucial step for
building Knowledge Bases (KBs) from large text datasets. This is a
fundamental goal of IBM Research: build intelligent systems to collect,
organize, and process information efficiently.

The opposite step of graph-to-text (G2T) transfer is key in presenting
the data encapsulated in a knowledge graph into a text form, more easily
readable by humans like us.

Figure 1 gives an example of this bi-directional transfer accomplished
by our system.

![](https://github.com/IBM/regen/blob/assets/imgs/generation_example.svg)

<p align="center">
<b>Figure 1.</b> An example of knowledge transfer where the first two sentences of
the abstract of our paper (Dognin, et al., 2021) on top are processed
through our ReGen models. First, a knowledge graph is constructed, then
it is used as input to generate a paragraph of text using our system (on
the right). Note that the generated paragraph captures the original
sentences content accurately.
</p>

The transfer from text-to-graph yields a graph representation of the
main facts of the input sentences. The subsequent graph-to-text
translation provides another paragraph, distinct from the original input
but covering its facts accurately.

This bi-directional transfer of knowledge is a key principle of the
Trusted AI Team at IBM Research where we develop tools to make AI more
explainable, fair, robust, private, and transparent.

# IBM Research Introduces ReGen at EMNLP 2021

"Reinforced Generation" is the focus of this new work being presented at
EMNLP\'21. Our team explored the use of Reinforcement Learning to
improve quality of text-to-graph and graph-to-text generation.
Reinforced Generation or *ReGen* allows to improve quality significantly
upon traditional techniques.

A version of our EMNLP\'21 paper can be found on arXiv
<https://arxiv.org/abs/2108.12472>.

Our team is composed of Pierre Dognin (Tech Lead), Inkit Padhi, Igor
Melnyk and Payel Das.

Code will be released in the companion GitHub repos
<https://github.com/IBM/regen>

# IBM Research\'s Approach

Our approach is composed of several conceptually important steps:

-   Generation tasks (text-to-graph, graph-to-text) are reframed as
    sequence to sequence (seq2seq) translation tasks.

-   "Graph linearization" turns graphs into sequence of edges our models
    can process easily.

-   Pretrained Language Models (PLMs) built on large amount of data,
    such as T5, are fine-tuned on both generation tasks.

-   Both generation tasks are cast into the Reinforcement Learning
    framework where a reward is attributed to the generated sequence
    given a ground truth.

Following this approach, we can build task-specialized, or hybrid models
allowing generation in both directions, as presented in Figure 2.

![](https://github.com/IBM/regen/blob/assets/imgs/models.svg)

<p align="center">
<b>Figure 2.</b> Specialized and hybrid models rely on the same losses for
fine-tuning. Specialized models are dedicated to a given generation
direction while hybrid models can handle both directions (graph-to-text,
text-to-graph).
</p>

In traditional approaches a model is trained by generating sequences
that are then scored against ground truth examples, usually using a
cross entropy (CE) loss to update the model parameters, as shown in
Figure 2.

Our approach follows a variant of the REINFORCE policy gradient method
(Williams, 1992) where the baseline is the reward of the model output
under greedy max generation. This is known as Self-Critical Sequence
Training (SCST) (Rennie, et al., 2017) where the model serves as its own
critic, as seen in Figure 3.

![](https://github.com/IBM/regen/blob/assets/imgs/regen_rl.svg)

<p align="center">
<b>Figure 3.</b> ReGen models are trained using Self Critical Sequence Training
which is a policy gradient method where the baseline is the reward of
the output of greedy-max generation p<sup>*</sup>, the model acting as its own
critic. p<sup>s</sup> is a sampling of our policy that allows for exploration
during training. The policy p is initialized to a large T5 PLM to ensure
stability.
</p>

A large Pretrained Language Model (such as T5) is used as a good
starting point for our policy. This is to enable stable training using
our policy gradient method. Rewards are modality dependent (graph, text)
and must not only capture the information content but also the structure
validity of the generated sequence \-- this is particularly important
for directed graphs which require very constrained structure.

## Examples of Generation

We provide two working examples of T2G and G2T generation. The examples
emphasize the benefits of using ReGen, our RL-based method, compared to
using traditional CE-based methods of fine-tuning. In Figure 4, for both
examples, the input sample is at the top. Below on the left, we provide
the ground truth (in gray), while generated outputs for ReGen-CE and
ReGen-RL are provided on the right in color (orange for CE, blue for
RL). We can see that for these two examples ReGen-RL allows a more
enriched, precise transfer.

![](https://github.com/IBM/regen/blob/assets/imgs/examples.png)

<p align="center">
<b>Figure 4.</b> Examples of generation for T2G and G2T with the difference
between traditional CE and RL (ReGen) model outputs. Each example has an
input sample at the top (text for T2G, graph for G2T), below to the left
in gray is the ground truth of the target domain (graph for T2G, text
for G2T), and below to the right is the generated output in color
(orange for CE, blue for RL) to emphasize the difference between
generation from CE and RL ReGen.
</p>

# IBM Research\'s Lead

We compared ReGen to the top systems of the WebNLG 2020 Challenge
[https://webnlg-challenge.loria.fr/challenge_2020](The%20transfer%20from%20text-to-graph%20yields%20a%20graph%20representation%20of%20the%20main%20facts%20of%20the%20input%20sentences.%20The%20subsequent%20graph-to-text%20translation%20provides%20another%20paragraph,%20distinct%20from%20the%20original%20input%20but%20covering%20the%20facts.),
a well-regarded public challenge for multilingual bi-directional
generation between text and knowledge graph.

WebNLG is a difficult challenge. Its dataset is relatively small (13K
train, 1.7K dev, 1.8K test) and includes unseen categories at test time.
ReGen establishes new state-of-the-art results WebNLG 2020 Challenge
dataset by large margins for both text-to-graph and graph-to-text

Direction, as demonstrated in Table 1 and Table 2.

On the much larger dataset TekGen (6.3M train, 5Kdev, 50K test), ReGen
shows consistent gains for using Reinforced Generation, validating its
use for large data operating points, as shown in Table 3 and Table 4.

We present results for both datasets, using well established metrics
such as BLEU, METEOR, chrF++ for text generation. For graph generation,
we use F1, Precision, Recall for nodes and edges w/ different levels of
matching (exact, partial, strict, entity type) as defined by the WebNLG
2020 Challenge. Note we only report results for exact match in Table 3
and Table 4, full results are in our paper (Dognin, et al., 2021) online
on arXiv <https://arxiv.org/abs/2108.12472>

 WebNLG G2T                             | BLEU↑    | BLEU NLTK↑ | METEOR↑ | chrF++↑
:---------------------------------------|:--------:|:----------:|:-------:|:-------:
Amazon AI (Shanghai) (Guo, et al.,2020) | 0.540    | 0.535      | 0.417   | 0.690 |
OSU Neural NLG (Li, et al., 2020)       | 0.535    | 0.532      | 0.414   | 0.688 |
Facebook FBConvAI (Yang, et al.,2020)   | 0.527    | 0.523      | 0.413   | 0.686 |
Google bt5 (Agarwal, et al., 2020)      | 0.517    | 0.517      | 0.411   | 0.679 |
IBM Research ReGen-CE                   | 0.553    | 0.549      | 0.418   | 0.694 |
IBM Research ReGen-RL (Dognin, et al. 2021) | **0.563** | **0.559** | **0.425** | **0.706** |

<p align="center">
<b>Table 1.</b> G2T best results for WebNLG 2020 Challenge dataset. The first
4 rows were the Challenge top performers. Results for IBM Research ReGen
CE and RL systems show gains from using Reinforcement Learning. Our
ReGen-RL is the best system overall, fine-tuning a t5-large model using
METEOR reward.
</p>


WebNLG T2G                                   | F1↑     | Precision↑     | Recall↑
:--------------------------------------------|:-------:|:--------------:|:-------:
Amazon AI (Shanghai) (Guo, et al., 2020)     | 0.689   | 0.689   | 0.690   |
Google bt5 (Agarwal, et al., 2020)           | 0.682   | 0.670   | 0.701   |
IBM Research ReGen-CE (Dognin, et al., 2021) | **0.723** | **0.714** | **0.738** |
IBM Research ReGen-RL (Dognin, et al., 2021) | 0.720   | 0.712   | 0.734   |

<p align="center">
<b>Table 2.</b> T2G best results for WebNLG 2020 Challenge dataset. The top 2
rows were the Challenge top performers. ReGen models improve upon all
metrics for all matching schemes, providing new State-of-the-art
results.
</p>


TekGen G2T                          |  BLEU↑ | BLEU NLTK↑ |  METEOR↑  |  chrF++↑
:-----------------------------------|:------:|:----------:|:---------:|:---------:
IBM ReGen-CE (Dognin, et al., 2021) |  0.241 |   0.242    |   0.233   |   0.405
IBM ReGen-RL (Dognin, et al., 2021) | **0.262** |  **0.262** |  **0.242** |  **0.422**


<p align="center">
<b>Table 3.</b> G2T TekGen Results: IBM Research ReGen-CE establishes a
  baseline on the large TekGen dataset. ReGen-RL consistently improves
  upon this baseline on all metrics for text-to-graph generation.
</p>


TekGen T2G                          | F1↑    | Precision↑ | Recall↑
:-----------------------------------|:------:|:----------:|:---------:
IBM ReGen-CE (Dognin, et al., 2021) |  0.619 |   0.605    |   0.643
IBM ReGen-RL (Dognin, et al., 2021) | **0.623** | **0.610** | **0.647**

<p align="center">
<b> Table 4. </b> T2G TekGen Results: IBM Research ReGen-CE establishes a
  baseline on the large TekGen dataset. ReGen-RL improves results on the
  test set compared to ReGen-CE on all metrics for text-to-graph
  generation.
</p>

# Future Work

Multiple exciting directions of research can now be explored given our
current work:

1.  Very large graph construction from large datasets of text is the
    ultimate goal for this research and ReGen is one step forward in
    that direction.

1.  Reward definition can allow for constrained generation in terms of
    both structure and content, which can be very beneficial for
    applications where constrained generated output is required.

1.  Fairness and Trust is another angle of investigation in this
    paradigm for both generation directions as starting point PLMs may
    display bias from its own training data.

# Details

A version of our EMNLP\'21 paper can be found online at
<https://arxiv.org/abs/2108.12472>.

Code will be released in the companion GitHub repos
<https://github.com/IBM/regen>

IBM Researchers involved with this work are Pierre Dognin (Tech Lead),
Inkit Padhi, Igor Melnyk, and Payel Das.

# Bibliography

Agarwal, O. et al., 2020. *Machine Translation Aided Bilingual
Data-to-Text Generation and Semantic Parsing.* Dublin, Ireland
(Virtual)

Dognin, P. L., Padhi, I., Melnyk, I. & Das, P., 2021. *ReGen:
Reinforcement Learning for Text and Knowledge Base Generation using
Pretrained Language Models.* Punta Cana, Dominican Republic, EMNLP\'21.

Guo, Q. et al., 2020. *P2: A Plan-and-Pretrain Approach for Knowledge
Graph-to-Text Generation.* Dublin, Ireland (Virtual), s.n.

Li, X., Maskharashvili, A., Jory Stevens-Guille, S. & White, M., 2020.
*Leveraging Large Pretrained Models for WebNLG 2020.* Dublin, Ireland
(Virtual)

Rennie, S. J. et al., 2017. *Self-critical sequence training for image
captioning.* Honolulu, Hawaii, Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition.

Williams, R. J., 1992. Simple statistical gradient-following algorithms
for connectionist reinforcement learning. *Machine learning,* 8(3), pp.
229-256.

Yang, Z. et al., 2020. *Improving Text-to-Text Pre-trained Models for
the Graph-to-Text Task.* Dublin, Ireland (Virtual)
