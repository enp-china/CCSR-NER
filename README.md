# Table of Contents

<details>

<summary><b>Expand Table of Contents</b></summary><blockquote><p align="justify">

- [Getting Started](#getting-started)
- [Train the language model](#train-the-language-model)
- [Embeddings description](#embeddings-description)
- [Train the named entity recognition model](#train-the-named-entity-recognition-model)

</p></blockquote></details>

---

# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

You need to have python >= 3.7


```
flair==0.4.1
Cython==0.29.6
spacy==2.1.3
```


## Installing

You need to clone the project https://github.com/enp-china/CCSR-NER.git

```
git clone https://github.com/enp-china/CCSR-NER.git
```

# Train the language model

If you want to use the same data we used, download and unzip the `wiki7M.raw.bz2` archive from [Zenodo](https://zenodo.org/record/4109839) and go to [Char language model](#char-language-model) or [Language model with word segmentation information](#language-model-with-word-segmentation-information) depending on the language model you want. 
this archive is a sample wikipedia of about 7 million sentences. If you want to use the entire wikipedia, please follow the first two parts below.

## download the data

- Download https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
- Use WikiExtractor.py to extract the paragraphs. ( https://github.com/attardi/wikiextractor )
- Use a script to split the paragraphs into sentences.  ( https://blog.csdn.net/blmoistawinde/article/details/82379256 )

## Format the data

- Replace spaces with a specific token ( e.g : \ue001 )
- Depending on the language model you want, the data format should change

###  Char language model

- Separate all characters with spaces
- Go to [Split the data](#split-the-data)

###  Language model with word segmentation information

#### Supervised segmentation

- For the supervised segmentation we used Zpar ( https://www.sutd.edu.sg/cmsresource/faculty/yuezhang/zpar.html ) which should return data segmented into words separated by spaces

MSR and PKU data can be found here : http://sighan.cs.uchicago.edu/bakeoff2005/ ant the Chinese Treebank 9.0 here : https://catalog.ldc.upenn.edu/LDC2016T13

<details>
<summary><b>Usage details</b></summary><blockquote><p align="justify">

- Download Zpar at https://sourceforge.net/projects/zpar/files/0.7.5/
- Unzip the archive
- To compile ZPar, type `make` in the zpar source directory
- Then `make zpar.zh` and `make segmentor`
- Then train a model : You'll need a train file with one word segmented sentence by line and run `zpar/dist/segmentor/train <train-file> <model-file> <number of iterations>`
- Then you'll be able to segment new texts by running : `zpar/dist/segmentor/segmentor <model> [<input-file>] [<output-file>]` Where the input files to the segmentor are formatted as a sequence of Chinese characters without any word segmentation.
- More informations can be found in the doc folder : `zpar/doc/doc/segmentor.html` 

</p></blockquote></details>

Now if you want to train a word language model you can leave the data as it is.

If you want to train our char-seg language model you'll need to run the `python language_model/format_data_bies.py input_file output_file` script with as first parameter the data segmented into word and as second parameter the desired name for the formatted file.


#### Unsupervised segmentation

For the unsupervised segmentation we used eleve ( https://github.com/kodexlab/eleve ).
Once the package is pip-installed, you can use the `eleve-chinese` command line tool.
Remember to use the `-bies` option to obtain correctly formated data.

<details>
<summary><b>Usage details</b></summary><blockquote><p align="justify">

- To train a model : `eleve-chinese train --corpus data/wiki7M.raw --model model_name [--training_length 1000000]`
- To segment a file : `eleve-chinese segment --corpus data/wiki7M.raw --model model_name --target data/wiki7M.raw.seg --bies`

</p></blockquote></details>


## Split the data

You need to split your corpus into train, validation and test portions. Flair's trainer class assumes that there is a folder for the corpus in which there is a 'test.txt' and a 'valid.txt' with test and validation data. Importantly, there is also a folder called 'train' that contains the training data in splits. For instance, the billion word corpus is split into 100 parts. The splits are necessary if all the data does not fit into memory, in which case the trainer randomly iterates through all splits.

So, the folder structure must look like this:

> corpus/ <br>
corpus/train/ <br>
corpus/train/train_split_1 <br>
corpus/train/train_split_2 <br>
corpus/train/... <br>
corpus/train/train_split_X <br>
corpus/test.txt <br>
corpus/valid.txt <br>

## Training the language model

run `python language_model/train_LM.py path_to_data path_to_model is_forward_lm`

where :
- `path_to_data` is the path to your training data ( eg : corpus/ )
- `path_to_model` is the path where you want to store your model
- `is_forward_lm` is a boolean to train a forward language model ( if True ) or a backward model ( if False )

# Embeddings description


You can use two kind of embeddings , contextual ( from flair ) or classic one ( e.g a .vec from FastText )

### Flair

You can have two kind of flair embeddings :

- Contextual char embedding : the contextual LM must have ".char.pt" extention to be recognized
- Contextual char seg embedding : the contextual LM must have ".char-seg.pt" extention to be recognized

Note : If you are using Flair, we recommend using the combination of the two models ( forward & backward ). 

### Vector file

You can have three kind of embeddings :

- Char embedding : The file must have ".char.vec" extention to be recognized
- Char-seg embedding : The file must have ".char-seg.vec" extention to be recognized
- Bichar embedding : The file must have ".bichar.vec" extention to be recognized

In order to train the NER model you will then have to rename your embeddings according to how they were trained

# Train the named entity recognition model

## Format the data

The format of the data should be a BIO format :


> 吳 B-NAME <br>
重 I-NAME <br>
昜 E-NAME <br>
， O <br>
... <br>

Depending on what type of embeddings you want to use the format will have to change.

If you want to use only characters embeddings you can leave the format as it is.

If you want to use our char-seg embeddings you'll need to add the word segmentation information in your data to get something like this :

> 吳-B B-NAME <br>
重-I I-NAME <br>
昜-E E-NAME <br>
，-S O <br>
... <br>

To do so you'll need a word segmentation model. 


For the supervised segmentation we used Zpar ( https://www.sutd.edu.sg/cmsresource/faculty/yuezhang/zpar.html ) which should return a model after training on segmented data.

For the unsupervised segmentation we used eleve ( https://github.com/kodexlab/eleve ) which should return a model after training on unsegmented data.

Once you have your models, normally they are the same ones that allowed you to train the language model, you can run `python language_model/format_bio_data.py --input path_to_the_input_file --output path_to_the_output_file --model path_to_the_model --unsup True`


the arguments of this script are :

> --input : which is the path to the input file [mandatory]<br>
--output : which is the path to the output file [mandatory]<br>
--model : which is the path to the model file [mandatory]<br>
--unsup : which is a boolean to specify if you use a supervised or unsupervised model [preset to False]<br>
--zpar : Path to the segmentor of Zpar [mandatory if you use a supervised model] ( It is normally found at `zpar-0.7.5/dist/segmentor/segmentor`)<br>
--char_ind : Index of the char in your BIO file [preset to 0]<br>
--tag_ind : Index of the ner tag in your BIO file [preset to 1]<br>

## Create a training config

In order to train the model you must set up a configuration file like the example proposed at `training_config/train.yml`

## Training the ner model 

To train a model:

```
python src/trainer.py training_config/train.yml
```

## Authors

**Baptiste Blouin**

**Pierre Magistry**

