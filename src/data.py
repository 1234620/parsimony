"""
Data pipeline.

Two jobs:
  1. Supply a reasoning-dense synthetic corpus. At <=50M parameters, data
     quality dominates data quantity (cf. TinyStories, phi-series), and the
     GIBC rubric explicitly scores reasoning on logic/math tasks.
  2. Hold out validation as RAW TEXT, never as tokens. Perplexity-per-token
     is not comparable across tokenizers with different vocabularies, so the
     vocab study would be meaningless if we cached tokenized validation data.
     We keep bytes, and each tokenizer encodes them itself at eval time.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

NAMES = ["Mira", "Tomas", "Anya", "Ravi", "Elin", "Petra", "Kofi", "Suri", "Bram", "Noor",
         "Idris", "Wren", "Halle", "Oscar", "Yuki", "Dara", "Ines", "Marek", "Tessa", "Omar"]
OBJECTS = ["lantern", "kite", "basket", "compass", "seashell", "ledger", "kettle", "sketchbook",
           "marble", "umbrella", "flute", "telescope", "ribbon", "acorn", "mirror", "satchel"]
PLACES = ["the harbour", "the orchard", "the attic", "the river bend", "the market", "the lighthouse",
          "the quarry", "the greenhouse", "the rooftop", "the old library", "the dunes", "the workshop"]
ADJ = ["small", "rusted", "bright", "heavy", "folded", "quiet", "cracked", "warm", "pale", "worn"]



# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------
# The first version of this generator used ~60 hand-written nouns. BPE
# saturated at 753 merges, so every requested vocabulary above ~1k collapsed
# to the same tokenizer and the vocabulary study became degenerate. Real
# lexical diversity is therefore a *correctness requirement* for the
# experiment, not a nicety -- and it makes the reasoning traces better
# training data, since the model can no longer memorise a handful of templates.

_FALLBACK_LEX = NAMES + OBJECTS + ADJ + [
    "harbour", "orchard", "attic", "market", "lighthouse", "quarry", "rooftop",
]


def load_lexicon(path: "Path | str | None" = None) -> List[str]:
    for cand in [path, Path(__file__).resolve().parents[1] / "data" / "lexicon.txt",
                 Path("/usr/share/hunspell/en_US.dic")]:
        if cand is None:
            continue
        cand = Path(cand)
        if not cand.exists():
            continue
        if cand.suffix == ".dic":
            words = []
            with cand.open(encoding="latin-1") as f:
                next(f, None)
                for line in f:
                    w = line.split("/")[0].strip()
                    if w.isalpha() and w.islower() and 3 <= len(w) <= 12:
                        words.append(w)
            if words:
                return sorted(set(words))
        else:
            words = [w.strip() for w in cand.read_text(encoding="utf-8").split("\n") if w.strip()]
            if words:
                return words
    return list(_FALLBACK_LEX)


class SyntheticCorpus:
    """Procedural corpus: narrative + arithmetic + comparative-logic traces.

    The reasoning slices emit explicit step-by-step derivations, so the model
    is trained on the *process*, not only the answer -- the single highest-
    leverage data choice for reasoning at small scale.
    """

    def __init__(self, seed: int = 0, lexicon: Optional[List[str]] = None, use_lexicon: bool = True):
        self.rng = random.Random(seed)
        self.lex = (lexicon if lexicon is not None else load_lexicon()) if use_lexicon else None
        if self.lex and len(self.lex) > 400:
            r = random.Random(seed + 1)
            self.names = [w.capitalize() for w in r.sample(self.lex, 4000)]
            self.objects = r.sample(self.lex, 6000)
            self.places = r.sample(self.lex, 3000)
            self.adj = r.sample(self.lex, 3000)
            self.verbs = r.sample(self.lex, 2000)
        else:
            self.names, self.objects = list(NAMES), list(OBJECTS)
            self.places, self.adj = [p.replace("the ", "") for p in PLACES], list(ADJ)
            self.verbs = ["kept", "moved", "traded", "mended"]

        # Natural language is Zipfian: a few types account for most tokens.
        # Sampling the lexicon uniformly would make a large BPE vocabulary look
        # artificially useless, because there would be no frequent words for it
        # to capture. Weighting by 1/rank keeps the tokenizer study honest.
        self._cw = {}
        for key, pop in (("names", self.names), ("objects", self.objects),
                         ("places", self.places), ("adj", self.adj), ("verbs", self.verbs)):
            cum, acc = [], 0.0
            for i in range(len(pop)):
                acc += 1.0 / (i + 1.0)      # Zipf(s=1) over rank
                cum.append(acc)
            self._cw[key] = cum

    def _pick(self, key, pop):
        return self.rng.choices(pop, cum_weights=self._cw[key], k=1)[0]

    def _n(self):  return self._pick("names", self.names)
    def _o(self):  return self._pick("objects", self.objects)
    def _p(self):  return self._pick("places", self.places)
    def _a(self):  return self._pick("adj", self.adj)
    def _v(self):  return self._pick("verbs", self.verbs)

    # ---- narrative --------------------------------------------------------
    def story(self) -> str:
        r = self.rng
        who, friend = self._n(), self._n()
        obj, obj2, place, adj = self._o(), self._o(), self._p(), self._a()
        templates = [
            f"{who} found a {adj} {obj} near the {place}. Nobody knew who had left it there. "
            f"{who} showed the {obj} to {friend}, who wanted it for the {obj2}. "
            f"They {self._v()} it together and walked back past the {place}.",
            f"Every morning {who} walked to the {place} carrying a {adj} {obj}. "
            f"One day the {obj} was missing. {friend} had {self._v()} it by mistake, "
            f"thinking it belonged to the {obj2}. {who} was not angry for long.",
            f"The {place} was quiet until {who} arrived with a {obj}. "
            f"{friend} asked where the {adj} {obj2} had gone. "
            f"{who} said it was still at the {place}, and they went to look for it.",
            f"{who} and {friend} argued about the {adj} {obj}. "
            f"{friend} thought it came from the {place}; {who} was sure it came from the {obj2}. "
            f"Neither of them was right, and they laughed about it afterwards.",
        ]
        return r.choice(templates)

    # ---- arithmetic with worked steps ------------------------------------
    def arithmetic(self) -> str:
        r = self.rng
        who = self._n()
        item = self._o() + "s"
        a, b = r.randint(3, 40), r.randint(2, 20)
        kind = r.choice(["add", "sub", "mul", "share"])
        if kind == "add":
            c = r.randint(2, 15)
            total = a + b + c
            return (f"Question: {who} has {a} {item}, then gets {b} more, then finds {c} more. "
                    f"How many {item} does {who} have?\n"
                    f"Step 1: {a} + {b} = {a+b}.\nStep 2: {a+b} + {c} = {total}.\n"
                    f"Answer: {total}.")
        if kind == "sub":
            b = min(b, a)
            return (f"Question: {who} has {a} {item} and gives away {b}. How many are left?\n"
                    f"Step 1: {a} - {b} = {a-b}.\nAnswer: {a-b}.")
        if kind == "mul":
            a, b = r.randint(2, 12), r.randint(2, 12)
            return (f"Question: There are {a} boxes with {b} {item} in each. How many {item} in total?\n"
                    f"Step 1: {a} x {b} = {a*b}.\nAnswer: {a*b}.")
        n = r.randint(2, 6)
        total = n * r.randint(2, 12)
        return (f"Question: {who} shares {total} {item} equally among {n} friends. How many each?\n"
                f"Step 1: {total} / {n} = {total//n}.\nAnswer: {total//n}.")

    # ---- comparative / transitive logic ----------------------------------
    def logic(self) -> str:
        r = self.rng
        x, y, z = self._n(), self._n(), self._n()
        rel = r.choice(["taller", "older", "faster", "quieter"])
        if r.random() < 0.5:
            return (f"Question: {x} is {rel} than {y}. {y} is {rel} than {z}. Who is {rel[:-2] if rel.endswith('er') else rel}est?\n"
                    f"Step 1: {x} > {y}.\nStep 2: {y} > {z}.\nStep 3: so {x} > {z}.\nAnswer: {x}.")
        a, b = r.randint(2, 30), r.randint(2, 30)
        while a == b:
            b = r.randint(2, 30)
        hi = x if a > b else y
        return (f"Question: {x} has {a} coins. {y} has {b} coins. Who has more?\n"
                f"Step 1: compare {a} and {b}.\nStep 2: {max(a,b)} > {min(a,b)}.\nAnswer: {hi}.")

    def sample(self, mix=(0.5, 0.3, 0.2)) -> str:
        u = self.rng.random()
        if u < mix[0]:
            return self.story()
        if u < mix[0] + mix[1]:
            return self.arithmetic()
        return self.logic()

    def documents(self, n: int, mix=(0.5, 0.3, 0.2)) -> Iterator[str]:
        for _ in range(n):
            yield self.sample(mix)


def write_corpus(path: Path, n_docs: int, seed: int = 0, mix=(0.5, 0.3, 0.2)) -> Path:
    corpus = SyntheticCorpus(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in corpus.documents(n_docs, mix):
            f.write(doc.replace("\n", "\\n") + "\n")
    return path


def read_corpus(path: Path) -> List[str]:
    with Path(path).open(encoding="utf-8") as f:
        return [line.rstrip("\n").replace("\\n", "\n") for line in f]


@dataclass
class TokenizedDataset:
    """Flat token stream packed into fixed-length blocks for causal LM."""
    tokens: np.ndarray
    block_size: int

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.block_size)

    def batch(self, batch_size: int, rng: np.random.Generator):
        ix = rng.integers(0, len(self.tokens) - self.block_size - 1, size=batch_size)
        x = np.stack([self.tokens[i : i + self.block_size] for i in ix])
        y = np.stack([self.tokens[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.astype(np.int64), y.astype(np.int64)


def pack_tokens(token_lists: List[List[int]], eos_id: int) -> np.ndarray:
    out: List[int] = []
    for t in token_lists:
        out.extend(t)
        out.append(eos_id)
    return np.array(out, dtype=np.uint16 if max(out or [0]) < 65535 else np.int32)
