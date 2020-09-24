from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import Corpus, Sentence, Token, Dictionary
from typing import List, Dict, Union
from pathlib import Path
import logging
import re
import numpy as np
from random import shuffle

log = logging.getLogger("flair")

from flair.datasets import ColumnDataset,ColumnCorpus
from torch.utils.data import Dataset, random_split


def shuffleDoc(train):
    train_num = np.arange(0, len(train))
    shuffle(train_num)
    new_train = []
    for tn in train_num:
        new_train.append(train[tn])
    return new_train


def ColumnCorpusTrain(
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        tag_to_bioes=None,
        in_memory: bool = True,
        eval_part: int = 0,
        min_occur=0.10
):
    """
    Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.
    :param data_folder: base folder with the task data
    :param column_format: a map specifying the column format
    :param train_file: the name of the train file
    :param test_file: the name of the test file
    :param dev_file: the name of the dev file, if None, dev data is sampled from train
    :param tag_to_bioes: whether to convert to BIOES tagging scheme
    :return: a Corpus with annotated train, dev and test data
    """

    if eval_part < 0 or eval_part > 10:
        print("eval part must be in range 0-10")
        exit(0)

    if type(data_folder) == str:
        data_folder: Path = Path(data_folder)

    if train_file is not None:
        train_file = data_folder / train_file

    # get train data
    train = ColumnDataset(
        train_file,
        column_format,
        tag_to_bioes,
        in_memory=in_memory,
    )

    # read in test file if exists, otherwise sample 10% of train data as test dataset

    good = True

    while good:
        print("looking for split..")
        train = shuffleDoc(train)

        tab_good = []
        tab_train = []
        tab_test = []
        tab_dev = []
        for eval_part in range(0, 10):

            train_length = len(train)
            dev_size: int = round(train_length / 10)
            test_size: int = round(train_length / 10)
            start_dev = dev_size * eval_part

            print(dev_size, test_size, start_dev)

            dev = train[start_dev:start_dev + dev_size]
            if eval_part < 9:
                test = train[start_dev + dev_size:start_dev + dev_size + test_size]
                train_ = train[:start_dev] + train[start_dev + dev_size + test_size:]
            else:
                dev = train[start_dev:]
                test = train[:test_size]
                train_ = train[test_size:start_dev]

            tab_dev.append(dev)
            tab_test.append(test)
            tab_train.append(train_)

            test_count = 0
            for t in test:
                done = False
                for tok in t.tokens:
                    if tok.get_tag("ner") != "O":
                        done = True
                if done:
                    test_count += 1
            dev_count = 0
            for t in dev:
                done = False
                for tok in t.tokens:
                    if tok.get_tag("ner") != "O":
                        done = True
                if done:
                    dev_count += 1

            print(dev_count, test_count, min_occur * len(dev), min_occur * len(test))
            if dev_count >= min_occur * len(dev) and test_count >= min_occur * len(test):
                tab_good += [0]
            else:
                tab_good += [1]

        good = False
        for t in tab_good:
            if t == 1:
                good = True

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
    corpus = []
    for i in range(len(tab_train)):
        corpus.append(Corpus(tab_train[i], tab_dev[i], tab_test[i]))

    return corpus



def makeFormatGold(sentence: str):
    split_str = sentence.split()
    list_tok = []
    for token in split_str:
        if len(token) == 1:
            tok = token + "-_-S"
            list_tok.append(tok)
        elif len(token) > 1:
            tok = token[0] + "-_" + token[1:] + "-B"
            list_tok.append(tok)
            for i in range(1, len(token) - 1):
                tok = token[i] + "-" + token[:i] + "_" + token[i + 1 :] + "-I"
                list_tok.append(tok)
            tok = token[-1] + "-" + token[:-1] + "_-E"
            list_tok.append(tok)
    return list_tok


def make_corpus_format(corpus, segmenter_func, segmenter, save=False):
    import subprocess
    import time

    if save != False:
        from pathlib import Path

        path = Path(save)
        if not path.exists():
            path.mkdir()
    proc = subprocess.Popen(
        [segmenter_func, segmenter],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(10)

    for train in corpus.train:
        sentence = ""
        for s in train.tokens:
            sentence += s.text
        sentence += "\n"
        proc.stdin.write(sentence.encode())
        proc.stdin.flush()
        segmentation = proc.stdout.readline().decode()

        segmentation = makeFormatGold(segmentation.replace("\n", ""))
        for i in range(len(segmentation)):
            train.tokens[i].text = segmentation[i]

    if save != False:
        train_path = path / "train"
        out = open(train_path, "w", encoding="utf8")
        for train in corpus.train:
            for i in range(len(train.tokens)):
                ligne = (
                    train.tokens[i].text
                    + " "
                    + train.tokens[i].tags["ner"].value
                    + "\n"
                )
                out.write(ligne)
            out.write("\n")
        out.close()

    for dev in corpus.dev:
        sentence = ""
        for s in dev.tokens:
            sentence += s.text
        sentence += "\n"
        proc.stdin.write(sentence.encode())
        proc.stdin.flush()
        segmentation = proc.stdout.readline().decode()

        segmentation = makeFormatGold(segmentation.replace("\n", ""))

        for i in range(len(segmentation)):
            dev.tokens[i].text = segmentation[i]

    if save != False:
        dev_path = path / "dev"
        out = open(dev_path, "w", encoding="utf8")
        for dev in corpus.dev:
            for i in range(len(dev.tokens)):
                ligne = (
                    dev.tokens[i].text + " " + dev.tokens[i].tags["ner"].value + "\n"
                )
                out.write(ligne)
            out.write("\n")
        out.close()

    for test in corpus.test:
        sentence = ""
        for s in test.tokens:
            sentence += s.text
        sentence += "\n"
        proc.stdin.write(sentence.encode())
        proc.stdin.flush()
        segmentation = proc.stdout.readline().decode()

        segmentation = makeFormatGold(segmentation.replace("\n", ""))

        for i in range(len(segmentation)):
            test.tokens[i].text = segmentation[i]

    if save != False:
        test_path = path / "test"
        out = open(test_path, "w", encoding="utf8")
        for test in corpus.test:
            for i in range(len(test.tokens)):
                ligne = (
                    test.tokens[i].text + " " + test.tokens[i].tags["ner"].value + "\n"
                )
                out.write(ligne)
            out.write("\n")
        out.close()

    return corpus


class NLPTDFetcher(NLPTaskDataFetcher):
    @staticmethod
    def load_column_corpus(
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        tag_to_biloes=None,
        focus_on="all",
    ) -> Corpus:
        """
        Helper function to get a TaggedCorpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_biloes: whether to convert to BILOES tagging scheme
        :return: a TaggedCorpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith(".gz"):
                    continue
                if "train" in file_name and not "54019" in file_name:
                    train_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

            # if no test file is found, take any file with 'test' in name
            if test_file is None:
                for file in data_folder.iterdir():
                    file_name = file.name
                    if file_name.endswith(".gz"):
                        continue
                    if "test" in file_name:
                        test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        # get train and test data
        sentences_train: List[Sentence] = NLPTDFetcher.read_column_data(
            train_file, column_format, focus_on
        )

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            sentences_test: List[Sentence] = NLPTDFetcher.read_column_data(
                test_file, column_format, focus_on
            )
        else:
            sentences_test: List[Sentence] = [
                sentences_train[i]
                for i in NLPTaskDataFetcher.__sample(len(sentences_train), 0.1)
            ]
            sentences_train = [x for x in sentences_train if x not in sentences_test]

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            sentences_dev: List[Sentence] = NLPTDFetcher.read_column_data(
                dev_file, column_format, focus_on
            )
        else:
            sentences_dev: List[Sentence] = [
                sentences_train[i]
                for i in NLPTaskDataFetcher.__sample(len(sentences_train), 0.1)
            ]
            sentences_train = [x for x in sentences_train if x not in sentences_dev]

        if tag_to_biloes is not None:
            # convert tag scheme to iobes
            for sentence in sentences_train + sentences_test + sentences_dev:
                sentence: Sentence = sentence
                sentence.convert_tag_scheme(
                    tag_type=tag_to_biloes, target_scheme="iobes"
                )

            return Corpus(
                sentences_train, sentences_dev, sentences_test, name=data_folder.name
            )

    @staticmethod
    def read_column_data(
        path_to_column_file: Path,
        column_name_map: Dict[int, str],
        focus_on="all",
        infer_whitespace_after: bool = True,
    ):
        """
        Reads a file in column format and produces a list of Sentence with tokenlevel annotation as specified in the
        column_name_map. For instance, by passing "{0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}" as column_name_map you
        specify that the first column is the text (lexical value) of the token, the second the PoS tag, the third
        the chunk and the forth the NER tag.
        :param path_to_column_file: the path to the column file
        :param column_name_map: a map of column number to token annotation name
        :param infer_whitespace_after: if True, tries to infer whitespace_after field for Token
        :return: list of sentences
        """
        sentences: List[Sentence] = []

        try:
            lines: List[str] = open(
                str(path_to_column_file), encoding="utf-8"
            ).read().strip().split("\n")
        except:
            log.info(
                'UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(
                    path_to_column_file
                )
            )
            lines: List[str] = open(
                str(path_to_column_file), encoding="latin1"
            ).read().strip().split("\n")

        # most data sets have the token text in the first column, if not, pass 'text' as column
        text_column: int = 0
        for column in column_name_map:
            if column_name_map[column] == "text":
                text_column = column

        sentence: Sentence = Sentence()
        for line in lines:

            if line.startswith("#"):
                continue

            if line.strip().replace("﻿", "") == "":
                if len(sentence) > 0:
                    sentence.infer_space_after()
                    sentences.append(sentence)
                sentence: Sentence = Sentence()

            else:
                fields: List[str] = re.split("\s+", line)
                token = Token(fields[text_column])
                for column in column_name_map:
                    if len(fields) > column:
                        if column != text_column:
                            if focus_on == "all":
                                token.add_tag(column_name_map[column], fields[column])
                            else:
                                if fields[column] == "O":
                                    token.add_tag(
                                        column_name_map[column], fields[column]
                                    )
                                elif fields[column].split("-")[1] in focus_on:
                                    token.add_tag(
                                        column_name_map[column], fields[column]
                                    )
                                else:
                                    token.add_tag(column_name_map[column], "O")
                sentence.add_token(token)

        if len(sentence.tokens) > 0:
            sentence.infer_space_after()
            sentences.append(sentence)

        return sentences


def make_tag_dic(corpus, tag_type: str, use_w=False) -> Dictionary:
    # Make the tag dictionary
    tag_dictionary: Dictionary = Dictionary()
    if not use_w:
        tag_dictionary.add_item("O")
    for sentence in corpus.get_all_sentences():
        for token in sentence.tokens:
            token: Token = token
            tag_dictionary.add_item(token.get_tag(tag_type).value)
    tag_dictionary.add_item("<START>")
    tag_dictionary.add_item("<STOP>")
    return tag_dictionary


def generateNerEmbFromTrain(train, labels):
    dic = {}
    for t in train.tokens:
        try:
            dic[t.text[0]][labels.index(t.get_tag("ner").value)] += 1
        except:
            dic.update({t.text[0]: np.zeros(len(labels))})
            dic[t.text[0]][labels.index(t.get_tag("ner").value)] = 1

    for d in dic:
        if np.sum(dic[d]) > 0:
            dic[d] = dic[d] / np.sum(dic[d])

    return dic
