import re
from abc import abstractmethod
import flair
from pathlib import Path
from typing import List, Union, Dict
import gensim
import numpy as np
import torch
from flair.data import Dictionary, Token, Sentence
from flair.embeddings import TokenEmbeddings
from flair.models import LanguageModel
from collections import Counter
from flair.data import Corpus

UNIFY =[]
do_uni=False

def get_unify_token(sentences):
    unify = []
    for sentence in sentences:
        sent = ""
        for t in sentence.tokens:
            if do_uni:
                sent+=return_unify_sino(t.text[0])
            else:
                sent+=t.text[0]
        unify.append(sent)

    return unify

def return_unify_sino(token):
    if do_uni:
        return UNIFY[token]
    else:
        return token


class CharacterEmbeddings(TokenEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(self, path_to_char_dict: str = None, embed_size: int = 25):
        """Uses the default character dictionary if none provided."""

        super().__init__()
        self.name = "Char"
        self.static_embeddings = False

        # use list of common characters if none provided
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load("common-chars")
        else:
            self.char_dictionary: Dictionary = Dictionary.load_from_file(
                path_to_char_dict
            )

        self.char_embedding_dim: int = embed_size
        self.hidden_size_char: int = embed_size
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary.item2idx), self.char_embedding_dim
        )
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim,
            self.hidden_size_char,
            num_layers=1,
            bidirectional=True,
        )

        self.__embedding_length = self.char_embedding_dim * 2

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        for sentence in sentences:

            tokens_char_indices = []

            # translate words in sentence into ints using dictionary
            for token in sentence.tokens:
                token: Token = token
                char_indices = [
                    self.char_dictionary.get_idx_for_item(return_unify_sino(char)) for char in token.text
                ]
                tokens_char_indices.append(char_indices)

            # sort words by length, for batching and masking
            tokens_sorted_by_length = sorted(
                tokens_char_indices, key=lambda p: len(p), reverse=True
            )
            d = {}
            for i, ci in enumerate(tokens_char_indices):
                for j, cj in enumerate(tokens_sorted_by_length):
                    if ci == cj:
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in tokens_sorted_by_length]
            longest_token_in_sentence = max(chars2_length)
            tokens_mask = torch.zeros(
                (len(tokens_sorted_by_length), longest_token_in_sentence),
                dtype=torch.long,
                device=flair.device,
            )

            for i, c in enumerate(tokens_sorted_by_length):
                tokens_mask[i, : chars2_length[i]] = torch.tensor(
                    c, dtype=torch.long, device=flair.device
                )

            # chars for rnn processing
            chars = tokens_mask

            character_embeddings = self.char_embedding(chars).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                character_embeddings, chars2_length
            )

            lstm_out, self.hidden = self.char_rnn(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.zeros(
                (outputs.size(0), outputs.size(2)),
                dtype=torch.float,
                device=flair.device,
            )
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[i, index - 1]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]

            for token_number, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, character_embeddings[token_number])

    def __str__(self):
        return self.name


class WordEmbeddingsVec(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(embeddings)
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            str(embeddings), binary=False
        )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = return_unify_sino(token.text)
                else:
                    word = return_unify_sino(token.get_tag(self.field).value)

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif (
                    re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif (
                    re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class WordEmbeddingsVecCharSeg(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(embeddings)
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            str(embeddings), binary=False
        )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word_split = word.split("-")
                if word_split[0] != "":
                    word = return_unify_sino(word_split[0]) + "-" + return_unify_sino(word_split[-1])
                else:
                    word = "--" + return_unify_sino(word_split[-1])

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif (
                    re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif (
                    re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class WordEmbeddingsVecWord(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None,position="all",use_uni=True):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(embeddings)
        self.static_embeddings = True
        self.position = position
        self.use_uni = use_uni
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            str(embeddings), binary=False
        )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word = word[2:-2].replace("_",word[0])

                if len(word) > 1 or self.use_uni:

                    if word in self.precomputed_word_embeddings:
                        word_embedding = self.precomputed_word_embeddings[word]
                    elif word.lower() in self.precomputed_word_embeddings:
                        word_embedding = self.precomputed_word_embeddings[word.lower()]
                    elif (
                            re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                    ):
                        word_embedding = self.precomputed_word_embeddings[
                            re.sub(r"\d", "#", word.lower())
                        ]
                    elif (
                            re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                    ):
                        word_embedding = self.precomputed_word_embeddings[
                            re.sub(r"\d", "0", word.lower())
                        ]
                    else:
                        word_embedding = np.zeros(self.embedding_length, dtype="float")

                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                if self.position =="last" and "-" in token.text[len(token.text)-3]:
                    word_embedding = torch.FloatTensor(word_embedding)
                elif self.position =="first" and "_" in token.text[2]:
                    word_embedding = torch.FloatTensor(word_embedding)
                elif self.position =="full":
                    word_embedding = torch.FloatTensor(word_embedding)
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class WordEmbeddingsBinWord(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None,position="all",use_uni=True):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(embeddings)
        self.static_embeddings = True
        self.position = position
        self.use_uni = use_uni
        self.precomputed_word_embeddings = gensim.models.FastText.load_fasttext_format(
            str(embeddings)
        )
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

        self.field = field

        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word = word[2:-2].replace("_", word[0])

                try :
                    if self.use_uni:
                        self.use_uni = True
                except:
                    self.use_uni = True


                if len(word) > 1 or self.use_uni:
                    if word in self.precomputed_word_embeddings:
                        word_embedding = self.precomputed_word_embeddings[word]
                    elif word.lower() in self.precomputed_word_embeddings:
                        word_embedding = self.precomputed_word_embeddings[word.lower()]
                    elif (
                        re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                    ):
                        word_embedding = self.precomputed_word_embeddings[
                            re.sub(r"\d", "#", word.lower())
                        ]
                    elif (
                        re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                    ):
                        word_embedding = self.precomputed_word_embeddings[
                            re.sub(r"\d", "0", word.lower())
                        ]
                    else:
                        word_embedding = np.zeros(self.embedding_length, dtype="float")
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")
                try:
                    if self.position == "last" and "-" in token.text[len(token.text) - 3]:
                        word_embedding = torch.FloatTensor(word_embedding)
                    elif self.position == "first" and "_" in token.text[2]:
                        word_embedding = torch.FloatTensor(word_embedding)
                    elif self.position == "full":
                        word_embedding = torch.FloatTensor(word_embedding)
                    else:
                        word_embedding = np.zeros(self.embedding_length, dtype="float")
                except:
                    word_embedding = torch.FloatTensor(word_embedding)

                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class WordEmbeddingsBinBichar(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(embeddings)
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.FastText.load_fasttext_format(
            str(embeddings)
        )
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

        self.field = field

        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value


                word = word[0]

                word_next = ""
                if token_idx < len(sentence) - 1:
                    token_next = sentence.tokens[token_idx + 1]
                    if "field" not in self.__dict__ or self.field is None:
                        word_next = token_next.text
                    else:
                        word_next = token_next.get_tag(self.field).value

                    word_next = word_next[0]
                else:
                    word_next += "\ue013"

                word = word + word_next

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif (
                    re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif (
                    re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class OneHotEmbeddings(TokenEmbeddings):
    """One-hot encoded embeddings."""

    def __init__(
        self,
        corpus=Union[Corpus, List[Sentence]],
        field: str = "text",
        embedding_length: int = 300,
        min_freq: int = 3,
    ):

        super().__init__()
        self.name = "one-hot"
        self.static_embeddings = False
        self.min_freq = min_freq

        tokens = list(map((lambda s: s.tokens), corpus.train))
        tokens = [token for sublist in tokens for token in sublist]

        if field == "text":
            most_common = Counter(list(map((lambda t: t.text), tokens))).most_common()
        else:
            most_common = Counter(
                list(map((lambda t: t.get_tag(field)), tokens))
            ).most_common()

        tokens = []
        for token, freq in most_common:
            if freq < min_freq:
                break
            tokens.append(token)

        self.vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            self.vocab_dictionary.add_item(token)

        # max_tokens = 500
        self.__embedding_length = embedding_length

        # print(self.vocab_dictionary.idx2item)
        print(f"vocabulary size of {len(self.vocab_dictionary)}")

        # model architecture
        self.embedding_layer = torch.nn.Embedding(
            len(self.vocab_dictionary), self.__embedding_length
        )
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        one_hot_sentences = []
        for i, sentence in enumerate(sentences):
            context_idxs = [
                self.vocab_dictionary.get_idx_for_item(t.text) for t in sentence.tokens
            ]

            one_hot_sentences.extend(context_idxs)

        one_hot_sentences = torch.tensor(one_hot_sentences, dtype=torch.long).to(
            flair.device
        )

        embedded = self.embedding_layer.forward(one_hot_sentences)

        index = 0
        for sentence in sentences:
            for token in sentence:
                embedding = embedded[index]
                token.set_embedding(self.name, embedding)
                index += 1

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return "min_freq={}".format(self.min_freq)



class nerEmbedding(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: dict, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = "nerEmbeddings"
        self.static_embeddings = True
        self.precomputed_word_embeddings = embeddings

        self.field = field

        self.__embedding_length: int = len(embeddings[list(embeddings.keys()[0])])
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word = word[0]

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif (
                    re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif (
                    re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class WordEmbeddingsVecBichar(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(embeddings)
        self.static_embeddings = True
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            str(embeddings), binary=False
        )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value


                word = word[0]

                if token_idx < len(sentence) - 1:
                    token_next = sentence.tokens[token_idx + 1]
                    if "field" not in self.__dict__ or self.field is None:
                        word_next = token_next.text
                    else:
                        word_next = token_next.get_tag(self.field).value

                    word_next = word_next[0]
                else:
                    word_next = "\ue013"

                word = word + word_next

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif (
                    re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif (
                    re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name





class WordEmbeddingsVecWordChar(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings_word: str, embeddings_char, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(embeddings_word)
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            str(embeddings_word), binary=False
        )

        self.precomputed_char_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            str(embeddings_char), binary=False
        )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word = word[2:-2].replace("_",word[0])

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif (
                    re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif (
                    re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    # Char part
                    word_embedding = np.zeros(self.embedding_length, dtype="float")
                    for char in word:
                        if char in self.precomputed_char_embeddings:
                            char_embedding = self.precomputed_char_embeddings[char]
                        elif char.lower() in self.precomputed_char_embeddings:
                            char_embedding = self.precomputed_char_embeddings[
                                char.lower()
                            ]
                        elif (
                            re.sub(r"\d", "#", char.lower())
                            in self.precomputed_char_embeddings
                        ):
                            char_embedding = self.precomputed_char_embeddings[
                                re.sub(r"\d", "#", char.lower())
                            ]
                        elif (
                            re.sub(r"\d", "0", char.lower())
                            in self.precomputed_char_embeddings
                        ):
                            char_embedding = self.precomputed_char_embeddings[
                                re.sub(r"\d", "0", char.lower())
                            ]
                        else:
                            char_embedding = np.zeros(
                                self.embedding_length, dtype="float"
                            )

                        word_embedding += char_embedding
                    word_embedding /= len(word)

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class WordEmbeddingsVecFirst(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(embeddings)
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            str(embeddings), binary=False
        )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word = word[0]

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif (
                    re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif (
                    re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name




from flair.models import LanguageModel

class LanguageModelWordLevel(LanguageModel):
    @classmethod
    def load_language_model(cls, model_file: Union[Path, str]):
        state = torch.load(str(model_file), map_location=flair.device)

        model = LanguageModelWordLevel(
            state["dictionary"],
            state["is_forward_lm"],
            state["hidden_size"],
            state["nlayers"],
            state["embedding_size"],
            state["nout"],
            state["dropout"],
        )

        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return model

    def get_representation(self, strings: List[List[str]], chars_per_chunk: int = 512):

        longest = len(strings[0])
        chunks = []
        splice_begin = 0
        for splice_end in range(chars_per_chunk, longest, chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in strings])
            splice_begin = splice_end

        chunks.append([text[splice_begin:longest] for text in strings])
        hidden = self.init_hidden(len(chunks[0]))

        output_parts = []

        # push each chunk through the RNN language model

        for chunk in chunks:
            sequences_as_char_indices: List[List[int]] = []
            for string in chunk:
                char_indices = [
                    self.dictionary.get_idx_for_item(char) for char in string
                ]
                sequences_as_char_indices.append(char_indices)

            batch = torch.tensor(
                sequences_as_char_indices, dtype=torch.long, device=flair.device
            ).transpose(0, 1)

            prediction, rnn_output, hidden = self.forward(batch, hidden)

            output_parts.append(rnn_output)

            # concatenate all chunks to make final output
        output = torch.cat(output_parts)

        return output


class LanguageModelWord(LanguageModel):
    @classmethod
    def load_language_model(cls, model_file: Union[Path, str]):
        state = torch.load(str(model_file), map_location=flair.device)

        model = LanguageModelWord(
            state["dictionary"],
            state["is_forward_lm"],
            state["hidden_size"],
            state["nlayers"],
            state["embedding_size"],
            state["nout"],
            state["dropout"],
        )

        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return model

    def get_representation(self, strings: List[str], chars_per_chunk: int = 512):

        longest = len(strings[0])
        chunks = []
        splice_begin = 0
        for splice_end in range(chars_per_chunk, longest, chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in strings])
            splice_begin = splice_end

        chunks.append([text[splice_begin:longest] for text in strings])
        hidden = self.init_hidden(len(chunks[0]))

        output_parts = []

        # push each chunk through the RNN language model

        for chunk in chunks:
            sequences_as_char_indices: List[List[int]] = []
            for string in chunk:
                char_indices = [
                    self.dictionary.get_idx_for_item(char) for char in string
                ]
                sequences_as_char_indices.append(char_indices)

            batch = torch.tensor(
                sequences_as_char_indices, dtype=torch.long, device=flair.device
            ).transpose(0, 1)

            prediction, rnn_output, hidden = self.forward(batch, hidden)

            output_parts.append(rnn_output)

            # concatenate all chunks to make final output
        output = torch.cat(output_parts)

        return output


class LanguageModelChar(LanguageModel):
    @classmethod
    def load_language_model(cls, model_file: Union[Path, str]):
        state = torch.load(str(model_file), map_location=flair.device)

        model = LanguageModelChar(
            state["dictionary"],
            state["is_forward_lm"],
            state["hidden_size"],
            state["nlayers"],
            state["embedding_size"],
            state["nout"],
            state["dropout"],
        )

        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return model

    def get_representation(self, strings: List[str], chars_per_chunk: int = 512):
        sequences_as_char_indices = []
        for s in strings:
            char_indices = []
            for c in s:
                if c == " " or c == "\n":
                    char_indices.append(self.dictionary.get_idx_for_item(c))
                else:
                    char_indices.append(
                        self.dictionary.get_idx_for_item(c[0])
                    )

            sequences_as_char_indices.append(char_indices)

        hidden = self.init_hidden(len(strings))
        output_parts = []
        batch = torch.LongTensor(sequences_as_char_indices).transpose(0, 1)
        batch = batch.to(flair.device)

        prediction, rnn_output, hidden = self.forward(batch, hidden)
        rnn_output = rnn_output.detach()

        output_parts.append(rnn_output)

        # concatenate all chunks to make final output
        output = torch.cat(output_parts)
        return output



class FlairEmbeddingsWordLevelCharSeg(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(
        self,
        model: str,
        use_cache: bool = False,
        cache_directory: Path = None,
        chars_per_chunk: int = 512,
        extra_index : int = 0,
    ):
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'
                depending on which character language model is desired.
        :param use_cache: if set to False, will not write embeddings to file for later retrieval. this saves disk space but will
                not allow re-use of once computed embeddings that do not fit into memory
        :param cache_directory: if cache_directory is not set, the cache will be written to ~/.flair/embeddings. otherwise the cache
                is written to the provided directory.
        :param  chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster but requires
                more memory. Lower means slower but less memory.
        """
        super().__init__()

        cache_dir = Path("embeddings")

        if not Path(model).exists():
            raise ValueError(
                f'The given model "{model}" is not available or is not a valid path.'
            )

        self.name = str(model)
        self.static_embeddings = True
        self.extra_index = extra_index
        self.lm = LanguageModelWordLevel.load_language_model(model)

        self.is_forward_lm: bool = self.lm.is_forward_lm
        self.chars_per_chunk: int = chars_per_chunk

        # initialize cache if use_cache set
        self.cache = None
        if use_cache:
            cache_path = (
                Path(f"{self.name}-tmp-cache.sqllite")
                if not cache_directory
                else cache_directory / f"{self.name}-tmp-cache.sqllite"
            )
            from sqlitedict import SqliteDict

            self.cache = SqliteDict(str(cache_path), autocommit=True)

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("h"))
        dummy_sentence.add_token(Token("h"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

        # set to eval mode
        self.eval()

    def train(self, mode=True):
        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["cache"] = None
        return state

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # make compatible with serialized models
        if "chars_per_chunk" not in self.__dict__:
            self.chars_per_chunk = 512

        # if cache is used, try setting embeddings from cache first
        if "cache" in self.__dict__ and self.cache is not None:

            # try populating embeddings from cache
            all_embeddings_retrieved_from_cache: bool = True
            for sentence in sentences:
                key = sentence.to_tokenized_string()
                embeddings = self.cache.get(key)

                if not embeddings:
                    all_embeddings_retrieved_from_cache = False
                    break
                else:
                    for token, embedding in zip(sentence, embeddings):
                        token.set_embedding(self.name, torch.FloatTensor(embedding))

            if all_embeddings_retrieved_from_cache:
                return sentences

        start_marker = "\n"

        end_marker = " "
        token_sentences = [[return_unify_sino(c.text[0])+"-"+c.text[len(c.text)-1] for c in sentence.tokens] for sentence in sentences]
        with torch.no_grad():

            longest_character_sequence_in_batch: int = len(
                max(token_sentences, key=len)
            )

            sentences_padded: List[str] = []
            append_padded_sentence = sentences_padded.append

            extra_offset = len(start_marker)

            for sentence_text in token_sentences:
                pad_by = longest_character_sequence_in_batch - len(sentence_text)
                if self.is_forward_lm:

                    padded = [start_marker] + sentence_text
                    padded.append(end_marker)

                    for i in range(pad_by):
                        padded.append(" ")

                    append_padded_sentence(padded)
                else:
                    padded = [start_marker] + sentence_text[::-1]
                    padded.append(end_marker)

                    for i in range(pad_by):
                        padded.append(" ")
                    append_padded_sentence(padded)

            all_hidden_states_in_lm = self.lm.get_representation(
                sentences_padded, self.chars_per_chunk
            )

            for i, sentence in enumerate(sentences):
                sentence_text = token_sentences[i]
                offset_forward: int = extra_offset +self.extra_index
                offset_backward: int = len(sentence_text) + self.extra_index
                for j, token in enumerate(sentence.tokens):

                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    offset_forward+=1
                    offset_backward-=1


                    token.set_embedding(self.name, embedding.clone())

            all_hidden_states_in_lm = all_hidden_states_in_lm.detach()
            all_hidden_states_in_lm = None

        return sentences

    def __str__(self):
        return self.name




class FlairEmbeddingsChar(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(
        self,
        model: str,
        use_cache: bool = False,
        cache_directory: Path = None,
        chars_per_chunk: int = 512,
        extra_index:int=0,
    ):
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'
                depending on which character language model is desired.
        :param use_cache: if set to False, will not write embeddings to file for later retrieval. this saves disk space but will
                not allow re-use of once computed embeddings that do not fit into memory
        :param cache_directory: if cache_directory is not set, the cache will be written to ~/.flair/embeddings. otherwise the cache
                is written to the provided directory.
        :param  chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster but requires
                more memory. Lower means slower but less memory.
        """
        super().__init__()

        cache_dir = Path("embeddings")

        if not Path(model).exists():
            raise ValueError(
                f'The given model "{model}" is not available or is not a valid path.'
            )

        self.name = str(model)
        self.static_embeddings = True
        self.extra_index = extra_index
        self.lm = LanguageModelWord.load_language_model(model)

        self.is_forward_lm: bool = self.lm.is_forward_lm
        self.chars_per_chunk: int = chars_per_chunk

        # initialize cache if use_cache set
        self.cache = None
        if use_cache:
            cache_path = (
                Path(f"{self.name}-tmp-cache.sqllite")
                if not cache_directory
                else cache_directory / f"{self.name}-tmp-cache.sqllite"
            )
            from sqlitedict import SqliteDict

            self.cache = SqliteDict(str(cache_path), autocommit=True)

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("h-_h-h"))
        dummy_sentence.add_token(Token("h-h_-h"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

        # set to eval mode
        self.eval()

    def train(self, mode=True):
        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["cache"] = None
        return state

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # make compatible with serialized models
        if "chars_per_chunk" not in self.__dict__:
            self.chars_per_chunk = 512

        # if cache is used, try setting embeddings from cache first
        if "cache" in self.__dict__ and self.cache is not None:

            # try populating embeddings from cache
            all_embeddings_retrieved_from_cache: bool = True
            for sentence in sentences:
                key = sentence.to_tokenized_string()
                embeddings = self.cache.get(key)

                if not embeddings:
                    all_embeddings_retrieved_from_cache = False
                    break
                else:
                    for token, embedding in zip(sentence, embeddings):
                        token.set_embedding(self.name, torch.FloatTensor(embedding))

            if all_embeddings_retrieved_from_cache:
                return sentences

        start_marker = "\n"

        end_marker = " "
        token_sentences = get_unify_token(sentences)
        with torch.no_grad():

            longest_character_sequence_in_batch: int = len(
                max(token_sentences, key=len)
            )

            sentences_padded: List[str] = []
            append_padded_sentence = sentences_padded.append

            extra_offset = len(start_marker)

            for sentence_text in token_sentences:
                pad_by = longest_character_sequence_in_batch - len(sentence_text)
                if self.is_forward_lm:
                    padded = "{}{}{}{}".format(
                        start_marker, sentence_text, end_marker, pad_by * " "
                    )
                    append_padded_sentence(padded)
                else:
                    padded = "{}{}{}{}".format(
                        start_marker, sentence_text[::-1], end_marker, pad_by * " "
                    )
                    append_padded_sentence(padded)

            all_hidden_states_in_lm = self.lm.get_representation(
                sentences_padded, self.chars_per_chunk
            )

            for i, sentence in enumerate(sentences):
                sentence_text = token_sentences[i]
                offset_forward: int = extra_offset +self.extra_index
                offset_backward: int = len(sentence_text)+self.extra_index
                for j, token in enumerate(sentence.tokens):

                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    offset_forward+=1
                    offset_backward-=1

                    token.set_embedding(self.name, embedding.clone())

            all_hidden_states_in_lm = all_hidden_states_in_lm.detach()
            all_hidden_states_in_lm = None

        return sentences

    def __str__(self):
        return self.name



class LanguageModelCharSeg(LanguageModel):
    @classmethod
    def load_language_model(cls, model_file: Union[Path, str]):
        state = torch.load(str(model_file), map_location=flair.device)

        model = LanguageModelCharSeg(
            state["dictionary"],
            state["is_forward_lm"],
            state["hidden_size"],
            state["nlayers"],
            state["embedding_size"],
            state["nout"],
            state["dropout"],
        )
        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return model

    def get_representation(self, strings: List[str], chars_per_chunk: int = 512):
        sequences_as_char_indices = []
        for s in strings:
            char_indices = []
            for c in s:
                if c == " " or c == "\n":
                    char_indices.append(self.dictionary.get_idx_for_item(c))
                elif c.split("-")[0] != "":
                    char_indices.append(
                        self.dictionary.get_idx_for_item(
                            c.split("-")[0] + "-" + c.split("-")[-1]
                        )
                    )
                else:
                    char_indices.append(
                        self.dictionary.get_idx_for_item("-" + "-" + c.split("-")[-1])
                    )
            sequences_as_char_indices.append(char_indices)

        hidden = self.init_hidden(len(strings))
        output_parts = []
        batch = torch.LongTensor(sequences_as_char_indices).transpose(0, 1)
        batch = batch.to(flair.device)

        prediction, rnn_output, hidden = self.forward(batch, hidden)
        rnn_output = rnn_output.detach()

        output_parts.append(rnn_output)

        # concatenate all chunks to make final output
        output = torch.cat(output_parts)
        return output






def reverse_slicing(s):
    return s[::-1]



class StackedEmbeddingsNew(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings], detach: bool = True):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            self.add_module("list_embedding_{}".format(i), embedding)

        self.detach: bool = detach
        self.name: str = "Stack"
        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(
        self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True
    ):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embed(embedding, sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'


def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
    """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
    are non-static."""

    # if only one sentence is passed, convert to list of sentence
    if type(sentences) is Sentence:
        sentences = [sentences]

    if "flair" in self.name:

        everything_embedded: bool = True

        if self.embedding_type == "word-level":
            for sentence in sentences:
                for token in sentence.tokens:
                    # print(token.text)
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)
    else:
        everything_embedded: bool = True

        if self.embedding_type == "word-level":
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)

    return sentences



class BertEmbeddingsChinese(TokenEmbeddings):
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
    ):
        from pytorch_pretrained_bert import BertTokenizer, BertModel

        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - bert_config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
        self.model = BertModel.from_pretrained(bert_model_or_path)
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.name = str(bert_model_or_path)
        self.static_embeddings = True

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(
            self,
            unique_id,
            tokens,
            input_ids,
            input_mask,
            input_type_ids,
            token_subtoken_count,
        ):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(
        self, sentences, max_sequence_length: int
    ) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text[0])
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0 : (max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(
                BertEmbeddings.BertInputFeatures(
                    unique_id=sentence_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    token_subtoken_count=token_subtoken_count,
                )
            )

        return features

    def to_tokenized_string(self,s:Sentence) -> str:

        if s.tokenized is None:
            s.tokenized = " ".join([t.text[0] for t in s.tokens])

        return s.tokenized

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(self.to_tokenized_string(sentence))
                    for sentence in sentences
                ],
                key=len,
            )
        )

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            flair.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            flair.device
        )

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        self.model.to(flair.device)
        self.model.eval()
        all_encoder_layers, _ = self.model(
            all_input_ids, token_type_ids=None, attention_mask=all_input_masks
        )

        with torch.no_grad():

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        layer_output = (
                            all_encoder_layers[int(layer_index)]
                            .detach()
                            .cpu()[sentence_index]
                        )
                        all_layers.append(layer_output[token_index])

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == "first":
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[
                            token_idx : token_idx
                            + feature.token_subtoken_count[token.idx]
                        ]
                        embeddings = [
                            embedding.unsqueeze(0) for embedding in embeddings
                        ]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return len(self.layer_indexes) * self.model.config.hidden_size





class BertEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
    ):
        from pytorch_pretrained_bert import BertTokenizer, BertModel

        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - bert_config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
        self.model = BertModel.from_pretrained(bert_model_or_path,from_tf=True)
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.name = str(bert_model_or_path)
        self.static_embeddings = True

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(
            self,
            unique_id,
            tokens,
            input_ids,
            input_mask,
            input_type_ids,
            token_subtoken_count,
        ):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(
        self, sentences, max_sequence_length: int
    ) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0 : (max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(
                BertEmbeddings.BertInputFeatures(
                    unique_id=sentence_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    token_subtoken_count=token_subtoken_count,
                )
            )

        return features

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(sentence.to_tokenized_string())
                    for sentence in sentences
                ],
                key=len,
            )
        )

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            flair.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            flair.device
        )

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        self.model.to(flair.device)
        self.model.eval()
        all_encoder_layers, _ = self.model(
            all_input_ids, token_type_ids=None, attention_mask=all_input_masks
        )

        with torch.no_grad():

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        layer_output = (
                            all_encoder_layers[int(layer_index)]
                            .detach()
                            .cpu()[sentence_index]
                        )
                        all_layers.append(layer_output[token_index])

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == "first":
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[
                            token_idx : token_idx
                            + feature.token_subtoken_count[token.idx]
                        ]
                        embeddings = [
                            embedding.unsqueeze(0) for embedding in embeddings
                        ]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return len(self.layer_indexes) * self.model.config.hidden_size


