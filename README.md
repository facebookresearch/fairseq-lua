# Introduction

***Note***: there is now a PyTorch version of this toolkit ([fairseq-py](https://github.com/pytorch/fairseq)) and new development efforts will focus on it. The Lua version is preserved here, but is provided without any support.

This is fairseq, a sequence-to-sequence learning toolkit for [Torch](http://torch.ch/) from Facebook AI Research tailored to Neural Machine Translation (NMT).
It implements the convolutional NMT models proposed in [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) and [A Convolutional Encoder Model for Neural Machine Translation](https://arxiv.org/abs/1611.02344) as well as a standard LSTM-based model.
It features multi-GPU training on a single machine as well as fast beam search generation on both CPU and GPU.
We provide pre-trained models for English to French, English to German and English to Romanian translation.

![Model](fairseq.gif)

# Citation

If you use the code in your paper, then please cite it as:

```
@article{gehring2017convs2s,
  author          = {Gehring, Jonas and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  title           = "{Convolutional Sequence to Sequence Learning}",
  journal         = {ArXiv e-prints},
  archivePrefix   = "arXiv",
  eprinttype      = {arxiv},
  eprint          = {1705.03122},
  primaryClass    = "cs.CL",
  keywords        = {Computer Science - Computation and Language},
  year            = 2017,
  month           = May,
}
```

and

```
@article{gehring2016convenc,
  author          = {Gehring, Jonas and Auli, Michael and Grangier, David and Dauphin, Yann N},
  title           = "{A Convolutional Encoder Model for Neural Machine Translation}",
  journal         = {ArXiv e-prints},
  archivePrefix   = "arXiv",
  eprinttype      = {arxiv},
  eprint          = {1611.02344},
  primaryClass    = "cs.CL",
  keywords        = {Computer Science - Computation and Language},
  year            = 2016,
  month           = Nov,
}
```

# Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* A [Torch installation](http://torch.ch/docs/getting-started.html). For maximum speed, we recommend using LuaJIT and [Intel MKL](https://software.intel.com/en-us/intel-mkl).
* A recent version [nn](https://github.com/torch/nn). The minimum required version is from May 5th, 2017. A simple `luarocks install nn` is sufficient to update your locally installed version.

Install fairseq by cloning the GitHub repository and running
```
luarocks make rocks/fairseq-scm-1.rockspec
```
LuaRocks will fetch and build any additional dependencies that may be missing.
In order to install the CPU-only version (which is only useful for translating new data with an existing model), do
```
luarocks make rocks/fairseq-cpu-scm-1.rockspec
```

The LuaRocks installation provides a command-line tool that includes the following functionality:
* `fairseq preprocess`: Data pre-processing: build vocabularies and binarize training data
* `fairseq train`: Train a new model on one or multiple GPUs
* `fairseq generate`: Translate pre-processed data with a trained model
* `fairseq generate-lines`: Translate raw text with a trained model
* `fairseq score`: BLEU scoring of generated translations against reference translations
* `fairseq tofloat`: Convert a trained model to a CPU model
* `fairseq optimize-fconv`: Optimize a fully convolutional model for generation. This can also be achieved by passing the `-fconvfast` flag to the generation scripts.

# Quick Start

## Training a New Model

### Data Pre-processing
The fairseq source distribution contains an example pre-processing script for
the IWSLT14 German-English corpus.
Pre-process and binarize the data as follows:
```
$ cd data/
$ bash prepare-iwslt14.sh
$ cd ..
$ TEXT=data/iwslt14.tokenized.de-en
$ fairseq preprocess -sourcelang de -targetlang en \
  -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/test \
  -thresholdsrc 3 -thresholdtgt 3 -destdir data-bin/iwslt14.tokenized.de-en
```
This will write binarized data that can be used for model training to data-bin/iwslt14.tokenized.de-en.

### Training
Use `fairseq train` to train a new model.
Here a few example settings that work well for the IWSLT14 dataset:
```
# Standard bi-directional LSTM model
$ mkdir -p trainings/blstm
$ fairseq train -sourcelang de -targetlang en -datadir data-bin/iwslt14.tokenized.de-en \
  -model blstm -nhid 512 -dropout 0.2 -dropout_hid 0 -optim adam -lr 0.0003125 -savedir trainings/blstm

# Fully convolutional sequence-to-sequence model
$ mkdir -p trainings/fconv
$ fairseq train -sourcelang de -targetlang en -datadir data-bin/iwslt14.tokenized.de-en \
  -model fconv -nenclayer 4 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
  -momentum 0.99 -timeavg -bptt 0 -savedir trainings/fconv

# Convolutional encoder, LSTM decoder
$ mkdir -p trainings/convenc
$ fairseq train -sourcelang de -targetlang en -datadir data-bin/iwslt14.tokenized.de-en \
  -model conv -nenclayer 6 -dropout 0.2 -dropout_hid 0 -savedir trainings/convenc
```

By default, `fairseq train` will use all available GPUs on your machine.
Use the [CUDA_VISIBLE_DEVICES](http://acceleware.com/blog/cudavisibledevices-masking-gpus) environment variable to select specific GPUs or `-ngpus` to change the number of GPU devices that will be used.

### Generation
Once your model is trained, you can translate with it using `fairseq generate` (for binarized data) or `fairseq generate-lines` (for text).
Here, we'll do it for a fully convolutional model:
```
# Optional: optimize for generation speed
$ fairseq optimize-fconv -input_model trainings/fconv/model_best.th7 -output_model trainings/fconv/model_best_opt.th7

# Translate some text
$ DATA=data-bin/iwslt14.tokenized.de-en
$ fairseq generate-lines -sourcedict $DATA/dict.de.th7 -targetdict $DATA/dict.en.th7 \
  -path trainings/fconv/model_best_opt.th7 -beam 10 -nbest 2
| [target] Dictionary: 24738 types
| [source] Dictionary: 35474 types
> eine sprache ist ausdruck des menschlichen geistes .
S	eine sprache ist ausdruck des menschlichen geistes .
O	eine sprache ist ausdruck des menschlichen geistes .
H	-0.23804219067097	a language is expression of human mind .
A	2 2 3 4 5 6 7 8 9
H	-0.23861141502857	a language is expression of the human mind .
A	2 2 3 4 5 7 6 7 9 9
```

### CPU Generation
Use `fairseq tofloat` to convert a trained model to use CPU-only operations (this has to be done on a GPU machine):
```
# Optional: optimize for generation speed
$ fairseq optimize-fconv -input_model trainings/fconv/model_best.th7 -output_model trainings/fconv/model_best_opt.th7

# Convert to float
$ fairseq tofloat -input_model trainings/fconv/model_best_opt.th7 \
  -output_model trainings/fconv/model_best_opt-float.th7

# Translate some text
$ fairseq generate-lines -sourcedict $DATA/dict.de.th7 -targetdict $DATA/dict.en.th7 \
  -path trainings/fconv/model_best_opt-float.th7 -beam 10 -nbest 2
> eine sprache ist ausdruck des menschlichen geistes .
S	eine sprache ist ausdruck des menschlichen geistes .
O	eine sprache ist ausdruck des menschlichen geistes .
H	-0.2380430996418	a language is expression of human mind .
A	2 2 3 4 5 6 7 8 9
H	-0.23861189186573	a language is expression of the human mind .
A	2 2 3 4 5 7 6 7 9 9
```

# Pre-trained Models

Generation with the binarized test sets can be run in batch mode as follows, e.g. for English-French on a GTX-1080ti:
```
$ fairseq generate -sourcelang en -targetlang fr -datadir data-bin/wmt14.en-fr -dataset newstest2014 \
  -path wmt14.en-fr.fconv-cuda/model.th7 -beam 5 -batchsize 128 | tee /tmp/gen.out
...
| Translated 3003 sentences (95451 tokens) in 136.3s (700.49 tokens/s)
| Timings: setup 0.1s (0.1%), encoder 1.9s (1.4%), decoder 108.9s (79.9%), search_results 0.0s (0.0%), search_prune 12.5s (9.2%)
| BLEU4 = 43.43, 68.2/49.2/37.4/28.8 (BP=0.996, ratio=1.004, sys_len=92087, ref_len=92448)

# Word-level BLEU scoring:
$ grep ^H /tmp/gen.out | cut -f3- | sed 's/@@ //g' > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- | sed 's/@@ //g' > /tmp/gen.out.ref
$ fairseq score -sys /tmp/gen.out.sys -ref /tmp/gen.out.ref
BLEU4 = 40.55, 67.6/46.5/34.0/25.3 (BP=1.000, ratio=0.998, sys_len=81369, ref_len=81194)
```

# Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users
* Contact: [jgehring@fb.com](mailto:jgehring@fb.com), [michaelauli@fb.com](mailto:michaelauli@fb.com)

# License
fairseq is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.
