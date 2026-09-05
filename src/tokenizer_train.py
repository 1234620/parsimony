"""Train byte-level BPE tokenizers at a target vocabulary size."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIALS = ["<pad>", "<eos>", "<unk>"]
EOS_ID = 1


def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: Path) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def corpus_stats(tok: Tokenizer, texts: List[str]) -> dict:
    """Compression stats -- the other half of the vocab tradeoff.

    A smaller vocabulary frees parameters but produces MORE tokens per byte,
    which costs compute and shortens effective context. Quantifying this is
    what makes the allocation study honest rather than cherry-picked.
    """
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tokens = sum(len(tok.encode(t).ids) for t in texts)
    return {
        "bytes": n_bytes,
        "tokens": n_tokens,
        "bytes_per_token": n_bytes / max(n_tokens, 1),
        "tokens_per_byte": n_tokens / max(n_bytes, 1),
        "actual_vocab": tok.get_vocab_size(),
    }
