# Self Attentive Sentence Embedding
This is the implementation for the paper **A Structured Self-Attentive Sentence Embedding**,  which is published in ICLR 2017: https://arxiv.org/abs/1703.03130 . We provide reproductions for the results on Yelp, Age and SNLI datasets, as well as their baselines. 

Thanks to the community, there have been various reimplementations of this work
by researchers from different groups before we release
this version of code. Some of them even achieved higher performances than the
results we reported in the paper. We would really like to thank them here, and refer
those third party implementations at the end of this readme. They provide
our model in different frameworks (TensorFlow, PyTorch) as well.


## Requirements:
[Theano](http://deeplearning.net/software/theano/)
[Lasagne](http://lasagne.readthedocs.io/en/latest/)
[scikit-learn](http://scikit-learn.org/stable/)
[NLTK](http://www.nltk.org/)


## Datasets and Preprocessing
The SNLI dataset can be downloaded from https://nlp.stanford.edu/projects/snli/ .
The file ``oov_vec.py`` is for preprocessing this dataset, no additional command line arguments needed.

For [Yelp](https://www.yelp.com/dataset_challenge) and [Age](http://pan.webis.de/clef16/pan16-web/author-profiling.html) data, they are preprocessed by the same file, with different command args:
```
oov_vec_nlc.py age2 glove
oov_vec_nlc.py yelp glove
```
You can also choose between `word2vec` and `glove` through the command line args.


## Word Embeddings
Our experiments are majorly based on GloVe embeddings (https://nlp.stanford.edu/projects/glove/), but we've also tested them on `word2vec` (https://code.google.com/archive/p/word2vec/) as well for Age and Yelp datasets.


## Traning Baselines
After running the preprocessing scripts beforehand, the baseline results on Age and Yelp datasets can be reproduced by the following configurations:

```
python lstmmlp_rate_l2_dpout.py 300 3000 0.06 0.0001 0.5 word2vec 100 16 0.5 300 0.1 1 age2
python lstmmlp_rate_l2_dpout.py 300 3000 0.06 0.0001 0.5 word2vec 100 32 0.5 300 0.1 1 yelp
```

## Training the Proposed Model

For reproducing the results in our paper on Age and Yelp, please run:
```
python semlp_rate_l2_dpout.py 300 350 2000 30 0.001 0.3 0.0001 1. glove 300 50 0.5 100 0.1 1 age2
python semlp_rate_l2_dpout.py 300 350 3000 30 0.001 0.3 0.0001 1. glove 300 50 0.5 100 0.1 1 yelp
```

And on SNLI dataset:
```
python segae_l2_dpout.py 300 150 3000 30 0.01 0.0001 0.3 1. 300 50 0.5 5 0.1 1
```

## Third Party Implementations
* PyTorch implementation by Haoyue Shi (@ExplorerFreda): https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding

* PyTorch implementation by Yufeng Ma (@yufengm): https://github.com/yufengm/SelfAttentive

* TensorFlow implementation by Diego Antognini (@Diego999): https://github.com/Diego999/SelfSent
