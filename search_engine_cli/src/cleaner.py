from multiprocessing.dummy import Array
from . import BASE_DIR, CORPUS_DIR

from nltk.corpus import reader

import os
import re
import glob


class Cleaner:
    """
    Base object for cleaning corpus located at ```./corpus```
    """

    def __init__(self, *args):
        self._strategies = list()
        self._current_doc = None

    def _with_nohtml(self):
        if self._current_doc:
            self._current_doc = re.sub(
                r'(?is)<(script|style).*?>.*?(</\1>)', "", self._current_doc.strip())
            self._current_doc = re.sub(
                r'(?s)<!--(.*?)-->[\n]?', "", self._current_doc)
            self._current_doc = re.sub(r'(?s)<.*?>', " ", self._current_doc)
            self._current_doc = re.sub(r'&nbsp;', " ", self._current_doc)

    def _with_moderate(self):
        # TODO:
        pass

    def _with_aggressive(self):
        # TODO:
        pass

    def strategy(self, s: str):
        for p in s.split("::"):
            self._strategies.append(p)
        return self

    def run(self):
        for src in glob.glob(f'{CORPUS_DIR}/*.*'):
            if not src.endswith('txt'):
                dst = os.path.splitext(src)[0]+'.txt'
                with open(src, 'r+') as f:
                    self._current_doc = f.read()
                    for p in self._strategies:
                        if hasattr(self, f'_with_{p}'):
                            getattr(self, f'_with_{p}')()
                    f.seek(0)
                    f.truncate()
                    f.write(" ".join(self._current_doc.split()))
                os.rename(src, dst)
        del self._current_doc
