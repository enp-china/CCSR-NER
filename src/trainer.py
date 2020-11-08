from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    BertEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
    ELMoEmbeddings,
    BytePairEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
)

#from flair.visual.training_curves import Plotter

import json
from flair.data import Corpus
from flair.visual.training_curves import Plotter
from .utils.trainer_utils import make_tag_dic, generateNerEmbFromTrain, ColumnCorpusTrain
import embeddings as emb
from pathlib import Path
from flair.models import SequenceTagger
from flair.optim import *
from typing import List
from flair.trainers.trainer import ModelTrainer
from flair.datasets import ColumnCorpus
import yaml

log = logging.getLogger("flair")


class trainer:
    def __init__(self, config: str):
        self.config = config

    def train_all(self):
        config_file = open(self.config, "r")
        if self.config.split('.')[-1] == "yml":
            datastore = yaml.load(config_file)
        elif self.config.split('.')[-1] == "json":
            datastore = json.loads(config_file.read())
        else:
            print("Need a json or yaml file as config")
            sys.exit(0)

        columns = {
            int(datastore["dataset_reader"]["position_text"]): "text",
            int(datastore["dataset_reader"]["position_ner"]): "ner",
        }

        # focus_on = datastore["dataset_reader"]["focus_on"]

        if bool(datastore["dataset_reader"]["only_train"]):

            all_corpus = []
            log.info("Reading data from {}".format(datastore["dataset_reader"]["data_folder"]))

            all_corpus = ColumnCorpusTrain(
                datastore["dataset_reader"]["data_folder"],
                columns,
                train_file=datastore["dataset_reader"]["train_name"],
            )

            tag_type = "ner"
            tag_dictionary = all_corpus[0].make_tag_dictionary(tag_type=tag_type)

        else:

            iobes_corpus = ColumnCorpus(
                datastore["dataset_reader"]["data_folder"],
                columns,
                train_file=datastore["dataset_reader"]["train_name"],
                dev_file=datastore["dataset_reader"]["dev_name"],
                test_file=datastore["dataset_reader"]["test_name"],
            )

            tag_type = "ner"
            tag_dictionary = iobes_corpus.make_tag_dictionary(tag_type=tag_type)

            try:
                train_ratio = float(datastore["dataset_reader"]["train_ratio"])
                iobes_corpus = Corpus(iobes_corpus.train[0:int(len(iobes_corpus.train) * train_ratio)],
                                      iobes_corpus.dev, iobes_corpus.test)
                log_ratio = "Using only ", str(train_ratio * 100), "% of the train dataset"
                log.info(log_ratio)
            except:
                pass

        embed_list = []
        word_char = []
        char_word = []
        for embed in datastore["embeddings"]["embeddings_list"]:

            if embed == "bpe":
                embed_list.append(BytePairEmbeddings(datastore["embeddings"]["lang"]))
            elif embed == "fasttext":
                embed_list.append(WordEmbeddings(datastore["embeddings"]["lang"]))
            elif embed == "flair" and datastore["embeddings"]["lang"] == "en":
                embed_list.append(FlairEmbeddings("news-forward"))
                embed_list.append(FlairEmbeddings("news-backward"))
            elif embed == "bert-base-uncased":
                if datastore["embeddings"]["lang"] == "en":
                    embed_list.append(BertEmbeddings("bert-base-uncased"))
            elif embed == "bert-base-cased":
                if datastore["embeddings"]["lang"] == "en":
                    embed_list.append(BertEmbeddings("bert-base-cased"))
            elif embed == "bert-large-uncased":
                if datastore["embeddings"]["lang"] == "en":
                    embed_list.append(BertEmbeddings("bert-large-uncased"))
            elif embed == "bert-large-cased":
                if datastore["embeddings"]["lang"] == "en":
                    embed_list.append(BertEmbeddings("bert-large-cased"))
            elif embed == "elmo-small":
                if datastore["embeddings"]["lang"] == "en":
                    embed_list.append(ELMoEmbeddings("small"))
            elif embed == "elmo-medium":
                if datastore["embeddings"]["lang"] == "en":
                    embed_list.append(ELMoEmbeddings("medium"))
            elif embed == "elmo-original":
                if datastore["embeddings"]["lang"] == "en":
                    embed_list.append(ELMoEmbeddings("original"))
            elif embed == "bert-base-chinese":
                if datastore["embeddings"]["lang"] == "zh":
                    embed_list.append(emb.BertEmbeddingsChinese("bert-base-chinese"))
            else:
                split_name = embed.split(".")
                ext = split_name[-1]
                kind = split_name[-2]

                if ext == "pt":  # Flair type

                    extra_index = 0
                    try:
                        extra_index = int(datastore["embeddings"]["extra_index"])
                    except:
                        pass

                    if kind == "char":
                        embed_list.append(emb.FlairEmbeddingsChar(embed, extra_index=extra_index))
                    elif kind == "char-seg":
                        embed_list.append(emb.FlairEmbeddingsWordLevelCharSeg(embed, extra_index=extra_index))

                if ext == "vec":  # Char type
                    if kind == "char-seg":
                        embed_list.append(emb.WordEmbeddingsVecCharSeg(embed))
                    elif kind == "char":
                        embed_list.append(emb.WordEmbeddingsVecFirst(embed))
                    elif kind == "word":
                        embed_list.append(emb.WordEmbeddingsVecWord(embed))
                    elif kind == "bichar":
                        embed_list.append(emb.WordEmbeddingsVecBichar(embed))
                if ext == "bin":
                    if kind == "word":
                        embed_list.append(emb.WordEmbeddingsBinWord(embed))
                    elif kind == "bichar":
                        embed_list.append(emb.WordEmbeddingsBinBichar(embed))

        try:
            if bool(datastore["embeddings"]["ner_embed"]) == True:
                print("Generate NER embeddings..")
                embed_list.append(
                    emb.nerEmbedding(
                        generateNerEmbFromTrain(
                            iobes_corpus.train, tag_dictionary.get_items()
                        )
                    )
                )
        except:
            pass
        try:
            if bool(datastore["embeddings"]["one_hot"]) == True:
                print("Generate one hot embeddings..")
                embed_list.append(emb.OneHotEmbeddings(iobes_corpus))
        except:
            pass
        try:
            if datastore["embeddings"]["embeddings_ngram_list"] != None:
                embed_list.append(
                    emb.WordEmbeddingsVecNGramList(
                        datastore["embeddings"]["embeddings_ngram_list"]
                    )
                )
        except:
            pass

        if len(word_char) == 1 and len(char_word) == 1:
            embed_list.append(emb.WordEmbeddingsVecWordChar(word_char[0], char_word[0]))

        embedding_types: List[TokenEmbeddings] = embed_list

        embeddings: emb.StackedEmbeddingsNew = emb.StackedEmbeddingsNew(
            embeddings=embedding_types
        )

        if bool(datastore["dataset_reader"]["only_train"]):
            score = []
            for i in range(len(all_corpus)):

                tagger: SequenceTagger = SequenceTagger(
                    hidden_size=int(datastore["model"]["hidden_size"]),
                    embeddings=embeddings,
                    tag_dictionary=tag_dictionary,
                    tag_type=tag_type,
                    use_crf=bool(datastore["model"]["use_crf"]),
                    dropout=float(datastore["model"]["dropout"]),
                    word_dropout=float(datastore["model"]["word_dropout"]),
                    locked_dropout=float(datastore["model"]["locked_dropout"]),
                    rnn_layers=int(datastore["model"]["rnn_layers"]),
                )

                folder = datastore["train_config"]["folder"] + "/" + str(i)
                best = Path(folder + "/checkpoint.pt")
                iobes_corpus = all_corpus[i]
                if not best.exists():
                    best = Path(folder + "/best-model.pt")

                if best.exists():
                    trainer = ModelTrainer.load_checkpoint(
                        tagger.load_checkpoint(best), iobes_corpus
                    )
                else:
                    trainer: ModelTrainer = ModelTrainer(tagger, iobes_corpus)

                # 7. start training

                result = trainer.train(
                    folder,
                    learning_rate=float(datastore["train_config"]["learning_rate"]),
                    anneal_factor=float(datastore["train_config"]["anneal_factor"]),
                    min_learning_rate=float(datastore["train_config"]["min_learning_rate"]),
                    mini_batch_size=int(datastore["train_config"]["batch_size"]),
                    max_epochs=int(datastore["train_config"]["epoch"]),
                    save_final_model=bool(datastore["train_config"]["save_final_model"]),
                    checkpoint=bool(datastore["train_config"]["checkpoint"]),
                    param_selection_mode=bool(
                        datastore["train_config"]["param_selection_mode"]
                    ),
                    patience=int(datastore["train_config"]["patience"]),
                    monitor_test=bool(datastore["train_config"]["monitor_test"]),
                    embeddings_storage_mode=str(datastore["train_config"]["embeddings_storage_mode"]),
                    shuffle=bool(datastore["train_config"]["shuffle"]),
                )

                plotter = Plotter()
                if bool(datastore["train_config"]["save_plot_training_curve"]):
                    curve = folder + "/loss.tsv"
                    plotter.plot_training_curves(curve)
                if bool(datastore["train_config"]["save_plot_weights"]):
                    weight = folder + "/weights.txt"
                    plotter.plot_weights(weight)

                score.append(result["test_score"])

            print(score, "  \n Moyenne : ", round(sum(score) / len(score), 2))


        else:

            tagger: SequenceTagger = SequenceTagger(
                hidden_size=int(datastore["model"]["hidden_size"]),
                embeddings=embeddings,
                tag_dictionary=tag_dictionary,
                tag_type=tag_type,
                use_crf=bool(datastore["model"]["use_crf"]),
                dropout=float(datastore["model"]["dropout"]),
                word_dropout=float(datastore["model"]["word_dropout"]),
                locked_dropout=float(datastore["model"]["locked_dropout"]),
                rnn_layers=int(datastore["model"]["rnn_layers"]),
            )

            folder = datastore["train_config"]["folder"]
            best = Path(folder + "/checkpoint.pt")
            if not best.exists():
                best = Path(folder + "/best-model.pt")

            if best.exists():
                trainer = ModelTrainer.load_checkpoint(
                    tagger.load_checkpoint(best), iobes_corpus
                )
            else:
                trainer: ModelTrainer = ModelTrainer(tagger, iobes_corpus)

            # 7. start training

            trainer.train(
                folder,
                learning_rate=float(datastore["train_config"]["learning_rate"]),
                anneal_factor=float(datastore["train_config"]["anneal_factor"]),
                min_learning_rate=float(datastore["train_config"]["min_learning_rate"]),
                mini_batch_size=int(datastore["train_config"]["batch_size"]),
                max_epochs=int(datastore["train_config"]["epoch"]),
                save_final_model=bool(datastore["train_config"]["save_final_model"]),
                checkpoint=bool(datastore["train_config"]["checkpoint"]),
                param_selection_mode=bool(
                    datastore["train_config"]["param_selection_mode"]
                ),
                patience=int(datastore["train_config"]["patience"]),
                monitor_test=bool(datastore["train_config"]["monitor_test"]),
                embeddings_storage_mode=str(datastore["train_config"]["embeddings_storage_mode"]),
                shuffle=bool(datastore["train_config"]["shuffle"]),
            )

            plotter = Plotter()
            if bool(datastore["train_config"]["save_plot_training_curve"]):
                curve = folder + "/loss.tsv"
                plotter.plot_training_curves(curve)
            if bool(datastore["train_config"]["save_plot_weights"]):
                weight = folder + "/weights.txt"
                plotter.plot_weights(weight)


import sys

if __name__ == "__main__":
    t = trainer(sys.argv[1])
    t.train_all()
