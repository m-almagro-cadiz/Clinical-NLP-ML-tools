# Clinical NLP & ML tools

This is a set of supervised tools and models to deal with a classification task in the Spanish clinical domain.

## Requirements

xgboost==1.0.2
unicode==2.6
snowballstemmer==1.2.1
scikit-learn==0.22.2.post1
pandas==0.25.3
numpy==1.18.1
networkx==2.1
nltk==3.2.5
python-Levenshtein==0.12.0
scipy==1.4.1
pdfminer.six==20200517
torch==1.9.0+cu102
torchvision==0.10.0+cu102
click==7.0
ruamel.yaml==0.16.5
joblib==0.13.2
logzero==1.5.0

## Datasets

* [CodiEsp](https://temu.bsc.es/codiesp/): spanish clinical notes annotated with icd-10.

## Components

* **Pre-processing pipeline**
* **BoW feature extraction module**
* **OvR wrapper**
* **HAN model**
* **RCNN model**
* **AttentionXML model**
* **Evaluation module**

## Examples

* OvR example
* HAN example 
* RCNN example
* AttentionXML example

## Reference
Lai et al., [Recurrent convolutional neural networks for text classification](https://dl.acm.org/doi/10.5555/2886521.2886636), Twenty-ninth AAAI conference on artificial intelligence 2015

Yang et al., [Hierarchical attention networks for document classification](https://aclanthology.org/N16-1174/), Proceedings of the 2016 conference of the North American 2016

You et al., [AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification](https://arxiv.org/abs/1811.01727), NeurIPS 2019

## Authors

* **Mario Almagro** - *AI Researcher* - [UNED](http://www.lsi.uned.es/)