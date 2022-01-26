from nis import match
from pydoc import doc
from . import (INDEX_DIR, CORPUS_DIR, STYLE_UNDERLINE,
               STYLE_BOLD, STYLE_GREEN, STYLE_CLEAR)

import re
import csv
import string
import math
import pprint
import timeit
import sys

from num2words import num2words
from operator import and_, or_, invert
from uuid import uuid4
from functools import reduce
from collections import Counter, OrderedDict, namedtuple

from nltk import download, word_tokenize, pos_tag, stem
from nltk.corpus import reader, stopwords, wordnet
from nltk.util import ngrams


# download NLTK data dependencies
download('punkt')
download('stopwords')
download('wordnet')
download('averaged_perceptron_tagger')

# maps to self._params tuple index
PREPROCESS_PARAMS_IDX = 0
PROCESS_PARAMS_IDX = 1
MODEL_PARAMS_IDX = 2
QUERY_PARAMS_IDX = 3


class Engine:
    """
    Base object for processing and indexing corpus located at ```./corpus```
    """

    def __init__(self, *args, **kwargs):
        t1, t2 = "Action", "Time(secs)"
        r = reader.PlaintextCorpusReader(CORPUS_DIR, ".*\.txt")

        self._params = ("lowercase", "nostopword", "boolean", None)
        self._corpus_refs = (dict(doc_id=uuid4().hex, file=field, txt=r.raw(
            field)) for field in r.fileids())
        self._doc_refs = list()
        self._results = tuple()
        # consist
        # :docid -> OrderedDict(id=doc_url)
        # :count -> OrderedDict(docid=int)
        # :vocab -> OrderedDict(token=tuple(doc))
        # :frequency -> tuple(tuple(docids), namedtuple(token=tuple(freq)))
        # :incidence -> tuple(tuple(docids), dict(token=binary_value))
        self._tables = dict()
        self._clean_build = True if kwargs.get(
            "clean_build") == None else kwargs.get("clean_build")
        print(f"\n{STYLE_BOLD}{STYLE_UNDERLINE}{t1:30s}{t2}{STYLE_CLEAR}" if self._clean_build else f"\n{STYLE_BOLD}{STYLE_GREEN} Using existing index!{STYLE_CLEAR}")

    def __index_path(self, suffix="post"):
        filename = '::'.join(self._params[:-1])
        path = f"{INDEX_DIR}/{filename}::{suffix}"
        return path

    def __lookup_docid(self, id: str):
        with open(self.__index_path("docid"), "r") as f:
            r = csv.reader(f)
            next(r)
            if id.strip() == "TO_DICT":
                return {row[0]: row[1] for row in r}
            if id.strip() == "ALL_IDS":
                return [row[0].strip() for row in r]
            return [row[1] for row in r if id.strip() == row[0].strip()][0]

    def __lookup_count(self, term: str):
        with open(self.__index_path("count"), "r") as f:
            r = csv.reader(f)
            next(r)
            if term.strip() == "TO_DICT":
                return {row[0]: int(row[1]) for row in r}
            return [row[1] for row in r if term.strip() == row[0].strip()][0]

    def __lookup_vocab(self, term: str):
        with open(self.__index_path("count"), "r") as f:
            r = csv.reader(f)
            next(r)
            if term.strip() == "TO_DICT":
                return {row[0]: int(row[1]) for row in r}
            return [row[1] for row in r if term.strip() == row[0].strip()][0]

    def __calc_tf_idf(self, term, freq) -> tuple:
        tmp = dict()
        if self._params[QUERY_PARAMS_IDX]:
            ids = tuple(self.__lookup_docid("ALL_IDS"))
            vocab = self.__lookup_docid("TO_DICT")
            count = self.__lookup_count("TO_DICT")
            n = len(ids)
            df = len(vocab.get(term) or [1])
            idf = math.log10(n/df)
            for id in ids:
                tf = freq/count.get(id)
                tf_idf = tf * idf
                tmp.update({id: tf_idf})
        else:
            vocab = self._tables.get("vocab")
            count = self._tables.get("count")
            ids = self._tables.get("frequency")[0]
            n = len(ids)
            df = len(vocab.get(term) or [1])
            idf = math.log10(n/df)
            for id in ids:
                tf = freq/count.get(id)
                tf_idf = tf + idf
                tmp.update({id: tf_idf})
        return tmp

    def __tokenise(self, s: str) -> tuple:
        return tuple(word_tokenize(s))

    def __unique_tokens(self, i: tuple) -> tuple:
        return tuple(set(i))

    def __count_tokens(self, i: tuple) -> Counter:
        return Counter(i)

    def __docid_table(self, *args, **kwargs):
        docid_table = {r.get("doc_id"): r.get("file") for r in self._doc_refs}
        self._tables.update(
            {"docid": OrderedDict(sorted(docid_table.items()))})
        with open(self.__index_path("docid"), 'w') as f:
            w = csv.writer(f)
            w.writerow(("id", "doc_name"))
            w.writerows([(id, doc_name)
                        for id, doc_name in self._tables.get("docid").items()])

    def __count_table(self, *args, **kwargs):
        count_table = {r.get("doc_id"): sum(r.get("count_tokens").values())
                       for r in self._doc_refs}
        self._tables.update(
            {"count": OrderedDict(sorted(count_table.items()))})
        with open(self.__index_path("count"), 'w') as f:
            w = csv.writer(f)
            w.writerow(("term", "count"))
            w.writerows([(id, count)
                        for id, count in self._tables.get("count").items()])

    def __vocab_table(self, *args, **kwargs):
        vocab_table = dict()
        for r in self._doc_refs:
            for t in r.get("unique_tokens"):
                if vocab_table.get(t):
                    vocab_table.update(
                        {t: vocab_table.get(t) + (r.get("doc_id"),)})
                else:
                    vocab_table.update(
                        {t: (r.get("doc_id"),)})
        self._tables.update(
            {"vocab": OrderedDict(sorted(vocab_table.items()))})
        with open(self.__index_path("vocab"), 'w') as f:
            w = csv.writer(f)
            w.writerow(("id", "doc_ids"))
            w.writerows([(id,) + doc_ids
                        for id, doc_ids in self._tables.get("vocab").items()])

    def __term_freq_matrix(self, *args, **kwargs):
        token_count = {doc_ref.get("doc_id"):  doc_ref.get("count_tokens")
                       for doc_ref in self._doc_refs}
        doc_ids = tuple(self._tables.get("docid").keys())
        tokens = tuple(self._tables.get("vocab").keys())
        freq_matrix = namedtuple("freq_matrix", tokens, rename=True)
        tmp = list()
        for token in tokens:
            row = [0]*len(doc_ids)
            for idx, id in enumerate(doc_ids):
                row[idx] = token_count.get(id).get(token) or 0
            tmp.append(tuple(row))
        self._tables.update({"frequency": (doc_ids, freq_matrix(*tmp))})

    def __incidence_matrix(self, *args, **kwargs):
        doc_ids = tuple(self._tables.get("docid").keys())
        vocabs = self._tables.get("vocab").items()
        token_row = dict()
        for token, docs in vocabs:
            r = list()
            for id in doc_ids:
                if id in docs:
                    r.append("1")
                else:
                    r.append("0")
            token_row.update({token: int("".join(r), 2)})
        self._tables.update({"incidence": (doc_ids, token_row)})

    def __gen_index(self):
        # order is important
        self.__debug_runtimer(self.__docid_table, msg="Docid Table")
        self.__debug_runtimer(self.__count_table, msg="Count Table")
        self.__debug_runtimer(self.__vocab_table, msg="Vocab Table")
        self.__debug_runtimer(self.__term_freq_matrix, msg="Frequency Matrix")
        self.__debug_runtimer(self.__incidence_matrix, msg="Incidence Matrix")

    def __gen_wordnet_pos(self, s: str) -> str:
        if s.startswith('J'):
            return wordnet.ADJ
        elif s.startswith('V'):
            return wordnet.VERB
        elif s.startswith('N'):
            return wordnet.NOUN
        elif s.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def _with_lowercase(self, *args, **kwargs):
        """
        Preprocess param
        """
        tmp = list()
        for r in self._corpus_refs:
            s = r.get('txt')
            tmp.append({**r, 'txt': s.lower()})
        self._corpus_refs = tuple(tmp)

    def _with_num2word(self, *args, **kwargs):
        """
        Preprocess param
        """
        tmp = list()
        for r in self._corpus_refs:
            s = r.get('txt')
            tmp.append({**r, 'txt': re.sub(r'[+-]?((\d+(\.\d+)?)|(\.\d+))',
                                           lambda g: num2words(s[g.span()[0]: g.span()[1]]), s)})
        self._corpus_refs = tuple(tmp)

    def _with_nopunctuation(self, *args, **kwargs):
        """
        Preprocess param
        """
        pun = string.punctuation
        tmp = list()
        for r in self._corpus_refs:
            s = r.get('txt')
            s = re.sub(r'[^A-Za-z0-9 ]+', '',
                       s.translate(str.maketrans(pun, " "*len(pun))))
            tmp.append({**r, 'txt': s})

        self._corpus_refs = tuple(tmp)

    def _with_nostopword(self, *args, **kwargs):
        """
        Process param
        """
        sw = set(stopwords.words('english'))
        tmp_a = list()
        for doc_ref in self._doc_refs:
            tmp_b = list()
            for tk in doc_ref.get("tokens"):
                if not tk in sw:
                    tmp_b.append(tk)
            tmp_a.append({**doc_ref, **{'tokens': tuple(tmp_b)}})
        self._doc_refs = tmp_a

    def _with_stem(self, *args, **kwargs):
        """
        Process param
        """
        tmp_a = list()
        for doc_ref in self._doc_refs:
            tmp_b = list()
            for tk in doc_ref.get("tokens"):
                tmp_b.append(stem.SnowballStemmer("english").stem(tk))
            tmp_a.append({**doc_ref, **{'tokens': tuple(tmp_b)}})
        self._doc_refs = tmp_a

    def _with_lemmatize(self, *args, **kwargs):
        """
        Process param
        """
        tmp_a = list()
        for doc_ref in self._doc_refs:
            tmp_b = list()
            for tk, tag in pos_tag(doc_ref.get("tokens")):
                _pos = self.__gen_wordnet_pos(tag)
                if _pos:
                    tmp_b.append(stem.WordNetLemmatizer().lemmatize(
                        tk, pos=_pos))
                else:
                    tmp_b.append(stem.WordNetLemmatizer().lemmatize(
                        tk))
            tmp_a.append({**doc_ref, **{'tokens': tuple(tmp_b)}})
        self._doc_refs = tmp_a

    def _with_bigram(self, *args, **kwargs):
        """
        Process param
        """
        tmp_a = list()
        for doc_ref in self._doc_refs:
            grams = tuple("".join(gram)
                          for gram in ngrams(doc_ref.get("tokens"), 2))
            tmp_b = doc_ref.get("tokens") + grams
            tmp_a.append({**doc_ref, **{'tokens': tmp_b}})
        self._doc_refs = tmp_a

    def _with_boolean(self, *args, **kwargs):
        """
        model param
        generates a boolean index `index/boolean`
        """
        fields, rows = self._tables.get("incidence")
        with open(self.__index_path(), 'w') as f:
            w = csv.writer(f)
            w.writerow(("term",) + fields)
            w.writerows([(term,) + tuple(str(bin(vecs))[2:].zfill(len(fields)))
                        for term, vecs in rows.items()])

    def _with_vector(self, *args, **kwargs):
        """
        model param
        generates a vector index `index/vector`
        """
        res = list()
        vecs = dict()
        terms = list()
        _, rows = self._tables.get("frequency")
        for term, freqs in rows._asdict().items():
            terms.append(term)
            res.append(self.__calc_tf_idf(term, sum(freqs)))
        for r in res:
            for doc_id, tf_idf in r.items():
                vecs.update(
                    {doc_id: (vecs.get(doc_id) if vecs.get(doc_id) else tuple()) + (tf_idf,)})
        with open(self.__index_path(), 'w') as f:
            w = csv.writer(f)
            w.writerow(("document", *terms))
            for k, v in vecs.items():
                w.writerow((k,) + v)

    def _use_boolean(self, *args, **kwargs):
        tmp = list()
        items = list()
        fields = None
        results = list()
        operators = re.sub(r'[^AND|OR|NOT]', " ",
                           self._params[QUERY_PARAMS_IDX]).split()
        queries = re.split(r'AND|OR|NOT', self._params[QUERY_PARAMS_IDX])
        for query in queries:
            self._corpus_refs = (
                dict(doc_id=uuid4().hex, file='query', txt=query.strip()),)
            self._doc_refs = list()
            self.preprocess()
            self.process()
            with open(self.__index_path(), "r") as f:
                r = csv.reader(f)
                fields = next(r)[1:]
                for row in r:
                    if row[0] in self._doc_refs[0]['unique_tokens']:
                        items.append(int("".join(row[1:]), 2))
                if not len(items):
                    items.append(0)
            tmp.append(reduce(and_, items))
        for op in operators:
            if op == "AND":
                tmp = [and_(*tmp[:2]), *tmp[2:]]
            elif op == "OR":
                tmp = [or_(*tmp[:2]), *tmp[2:]]
            elif op == "NOT":
                tmp[1] = invert(tmp[1])
                tmp = [and_(*tmp[:2]), *tmp[2:]]
        for idx, val in enumerate(str(bin(tmp[0]))[2:].zfill(len(fields))):
            if int(val):
                results.append(self.__lookup_docid(fields[idx]))
        self._results = tuple(results)

    def _use_vector(self, *args, **kwargs):
        results = list()
        vec1 = dict()
        vec2 = dict()
        query_tks = list()
        freqs = list()
        self._corpus_refs = (
            dict(doc_id=uuid4().hex, file='query', txt=self._params[QUERY_PARAMS_IDX].strip()),)
        self._doc_refs = list()
        self.preprocess()
        self.process()
        for term, freq in self._doc_refs[0].get("count_tokens").items():
            query_tks.append(term)
            freqs.append(freq)
            res = self.__calc_tf_idf(term, freq)
            for doc_id, tf_idf in res.items():
                vec2.update(
                    {doc_id: (vec2.get(doc_id) if vec2.get(doc_id) else tuple()) + (tf_idf,)})
        with open(self.__index_path(), "r") as f:
            r = csv.reader(f)
            corpus_tks = next(r)[1:]
            idx = list()
            for query_tk in query_tks:
                if query_tk in corpus_tks:
                    idx.append(corpus_tks.index(query_tk))
            for row in r:
                vec1.update({row[0]: tuple([row[i] for i in idx])})
        res = list()
        for doc_id in vec1.keys():
            v1 = vec1.get(doc_id)
            v2 = vec2.get(doc_id)
            m1 = math.sqrt(sum(float(v)**2 for v in v1))
            m2 = math.sqrt(sum(float(v)**2 for v in v2))
            n1 = [float(v)/m1 for v in v1]
            n2 = [float(v)/m2 for v in v2]
            dot = sum(_1 * _2 for _1, _2 in zip(n1, n2))
            res.append((doc_id, dot))
        res.sort(key=lambda arg: arg[1])
        for i in res:
            results.append(self.__lookup_docid(i[0]))
        self._results = tuple(results)

    def preprocess(self, s: str = None):
        val = (s := self._params[PREPROCESS_PARAMS_IDX] if not s else s)
        l = list(self._params)
        l[PREPROCESS_PARAMS_IDX] = val
        self._params = tuple(l)
        if self._clean_build:
            for p in s.split("::"):
                if hasattr(self, f'_with_{p}'):
                    self.__debug_runtimer(getattr(self, f'_with_{p}'), msg=p)
            for r in self._corpus_refs:
                self._doc_refs.append(dict(doc_id=r.get("doc_id"), file=r.get(
                    "file"), tokens=self.__tokenise(r.get('txt'))))
        return self

    def process(self, s: str = None):
        val = (s := self._params[PROCESS_PARAMS_IDX] if not s else s)
        l = list(self._params)
        l[PROCESS_PARAMS_IDX] = val
        self._params = tuple(l)
        if self._clean_build:
            for p in s.split("::"):
                if hasattr(self, f'_with_{p}'):
                    self.__debug_runtimer(getattr(self, f'_with_{p}'), msg=p)
            tmp = list()
            for doc_ref in self._doc_refs:
                tks = doc_ref.get("tokens")
                tmp.append({**doc_ref, 'unique_tokens': self.__unique_tokens(tks),
                            'count_tokens': self.__count_tokens(tks)})
            self._doc_refs = tmp
        return self

    def model(self, s: str = None):
        val = (s := self._params[MODEL_PARAMS_IDX] if not s else s)
        l = list(self._params)
        l[MODEL_PARAMS_IDX] = val
        self._params = tuple(l)
        if self._clean_build:
            self.__gen_index()
            for p in s.split("::"):
                if hasattr(self, f'_with_{p}'):
                    self.__debug_runtimer(getattr(self, f'_with_{p}'), msg=p)
        return self

    def query(self, s: str = None):
        self._clean_build = True
        del self._corpus_refs
        del self._doc_refs
        val = (s := self._params[QUERY_PARAMS_IDX] if not s else s)
        l = list(self._params)
        l[QUERY_PARAMS_IDX] = val
        self._params = tuple(l)
        for p in self._params[MODEL_PARAMS_IDX].split("::"):
            if hasattr(self, f'_use_{p}'):
                getattr(self, f'_use_{p}')()
        # clean up
        self._clean_build = False
        l = list(self._params)
        l[QUERY_PARAMS_IDX] = None
        self._params = tuple(l)
        return self._results

    def __debug_runtimer(self, func, *args, **kwargs):
        msg = kwargs.get("msg").capitalize() if kwargs.get("msg") else ""
        if self._params[QUERY_PARAMS_IDX]:
            print(f"\rsearching...\n", end='')
            ret = func(*args, **kwargs)
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
        elif self._clean_build:
            start = timeit.default_timer()
            print(f" -> applying {msg} ...")
            ret = func(*args, **kwargs)
            stop = timeit.default_timer()
            duration = stop - start
            duration = "{:.2f}".format(round(duration, 2))
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print(f"{msg:30s} {duration}")
        else:
            ret = func(*args, **kwargs)
        return ret
