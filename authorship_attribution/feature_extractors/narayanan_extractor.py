import string
import logging
from typing import List, Tuple
from itertools import product
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import tokenize
from nltk.parse.stanford import StanfordParser
from nltk.grammar import Nonterminal

from authorship_attribution.feature_extractors.base_extractor import BaseExtractor


class NarayananExtractor(BaseExtractor):
    def __init__(self):
        self.punctuations = ':;?.!,"' + "'"
        self.special_chars = r'#$%&\()*+/<=>@[\\]^_{|}~'
        self.language = "english"
        self.word_tokenizer = RegexpTokenizer(r'\w+')
        self.parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

        root = ["ROOT"]
        clauses = ["S", "SBAR", "SBARQ", "SINV", "SQ"]
        phrases = ["ADJP", "ADVP", "CONJP", "FRAG", "INTJ", "LST", "NAC", "NP", "NX", "PP", "PRN", "PRT", "QP", "RRC",
                   "UCP", "VP", "WHADJP", "WHAVP", "WHNP", "WHPP", "X"]
        words = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
                 "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
                 "VBZ", "WDT",
                 "WP", "WP$", "WRB"]

        self.parser_pairs = list(product(root, clauses)) + list(product(clauses, phrases)) + list(
            product(phrases, words))

    def transform(self, texts: List[str]) -> np.array:

        features = []
        for text in texts:
            row = [len(text.split())] + \
                  [self.character_count(text)] + \
                  [self.yules(text)] + \
                  self.punctuation_freq(text) + \
                  [text.count(x)/len(text) for x in self.special_chars] + \
                  list(self.character_frequency(text)) + \
                  [text.lower().count(char)/len(text) for char in string.digits]
            row += list(self.legomena(text)) + \
                   list(self.syntactic_pairs(text)) + \
                   self.word_shape(text) + \
                   self.stopword_frequency(text)  # function words

            features.append(np.array(row))  # single merged list is appended to list of lists
        return np.array(features)

    def average_characters_per_word(self, text: str) -> float:
        return self.character_count(text)/len(text.split())

    def stopword_frequency(self, text: str):
        words = [str(word).lower() for word in self.word_tokenizer.tokenize(text)]
        return [words.count(word)/len(words) for word in nltk.corpus.stopwords.words('english')]

    def syntactic_pairs(self, text: str):
        pairs = self.retrieve_pairs(text)
        return [pairs.count(pair)/len(pairs) for pair in self.parser_pairs]

    def punctuation_freq(self, text: str) -> List[float]:
        return [text.count(punctuation)/len(text) for punctuation in self.punctuations]

    def legomena(self, text: str) -> Tuple[float, float]:
        """
        hapax legomena	:	terms which occur only once in the corpus
        dis legomena	:	terms which occur twice in the corpus
        """

        freq = nltk.FreqDist(word.lower() for word in self.word_tokenizer.tokenize(text))
        if len(freq) == 0:
            logging.debug(text)
            return 0, 0

        hapax = [key for key, val in freq.items() if val == 1]
        dis = [key for key, val in freq.items() if val == 2]

        return len(hapax)/len(freq), len(dis)/len(freq)

    #pylint: disable=invalid-name
    def yules(self, text: str):
        """
        Returns a tuple with Yule's K and Yule's I.
        (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
        International Journal of Applied Linguistics, Vol 10 Issue 2)
        In production this needs exception handling.
        """
        token_counter = nltk.FreqDist(word.lower() for word in self.word_tokenizer.tokenize(text))
        m1 = sum(token_counter.values())
        m2 = sum([freq ** 2 for freq in token_counter.values()])
        if m1 == 0:
            logging.debug(text)
            return 0

        i = (m2 - m1)/(m1 * m1)
        k = 10000 * i
        return k

    @staticmethod
    def is_camel_case(s):
        if s != s.lower() and s != s.upper() and "_" not in s and sum(i.isupper() for i in s[1:-1]) == 1:
            return True
        return False

    def word_shape(self, text):
        words = self.word_tokenizer.tokenize(text)
        upper, lower, capitalized, camelcase, rest = 0, 0, 0, 0, 0
        distribution = np.zeros(20)
        inc = 1/len(words)

        for word in words:

            if len(word) < 21:
                distribution[len(word) - 1] += inc

            if str(word).isupper():
                upper += inc
            elif str(word).islower():
                lower += inc
            elif str(word)[0].isupper() and str(word)[1:].islower():
                capitalized += inc
            elif self.is_camel_case(str(word)):
                camelcase += inc
            else:
                rest += inc

        return [upper, lower, capitalized, camelcase, rest] + list(distribution)

    def retrieve_pairs(self, text):
        pairs = []
        for sentence in tokenize.sent_tokenize(text):
            try:
                tree = list(self.parser.raw_parse(sentence))[0]
                for production in tree.productions():
                    pairs.extend([(str(production._lhs), str(prod)) for prod in production._rhs if
                                  isinstance(prod, Nonterminal)])
            except Exception as e:
                logging.warning(e)
        return pairs

    @staticmethod
    def character_count(text: str) -> int:
        return len(text.replace(' ', ''))

    @staticmethod
    def character_frequency(text: str) -> Tuple:
        return tuple(text.lower().count(char) for char in string.ascii_lowercase)
