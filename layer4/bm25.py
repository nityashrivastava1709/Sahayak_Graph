"""
bm25.py  -  BM25, the classic keyword-ranking formula (what "conventional search" does). No dependencies.

For every query word that appears in a document, BM25 adds
      idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / average_length))
  idf  rare words count more           tf  how often the word is in the document (with diminishing returns)
  b    how strongly long documents are penalised   k1  how fast repeated words stop helping
Text is lower-cased, split into words, stop words are removed and plurals are reduced to a stem
("farmers" -> "farmer"). The stemmer is intentionally light.
"""
import math
import re
from collections import Counter

STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours
out over own same she should so some such than that the their theirs them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your yours""".split())


def stem(word: str) -> str:
    """Very light plural stemming: schemes -> scheme, families -> family, benefits -> benefit, class stays class."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list:
    words = re.findall(r"\w+", (text or "").lower())
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = 0
        self._tf, self._len, self._df, self._avg_len = [], [], Counter(), 0.0

    def fit(self, documents: list):
        """`documents` are raw texts. Returns self."""
        tokens = [tokenize(d) for d in documents]
        self.n_docs = len(tokens)
        self._tf = [Counter(t) for t in tokens]
        self._len = [len(t) for t in tokens]
        self._avg_len = (sum(self._len) / self.n_docs) if self.n_docs else 0.0
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        return self

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list:
        """One score per document (0.0 = no query word in the document)."""
        terms = list(dict.fromkeys(tokenize(query)))
        out = []
        for tf, length in zip(self._tf, self._len):
            s = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1.0))
                    s += self.idf(term) * f * (self.k1 + 1) / norm
            out.append(s)
        return out

    def top(self, query: str, k: int = 10) -> list:
        """[(document index, score)] best first, only documents with a positive score; ties keep input order."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
