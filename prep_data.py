"""C2G data preparation.

Source: enwik8 (English Wikipedia 100MB byte stream), canonical split
  train = bytes[0 : 90_000_000]
  valid = bytes[90_000_000 : 95_000_000]
  test  = bytes[95_000_000 : 100_000_000]

We use the *valid* head as our evaluation slice so that the evaluation set is
never seen during training and is byte-identical across every run.

Why enwik8 instead of FineWeb:
  HuggingFace (huggingface.co / cdn-lfs) is unreachable from this machine, so
  FineWeb cannot be downloaded. enwik8 is a public, fixed, byte-exact corpus
  with a standard split, which makes BPB here reproducible and comparable
  across our own ablation runs. Numbers are NOT comparable to the official
  FineWeb leaderboard and we never present them as such.

Tokenizer: byte-level BPE (GPT-2 style pre-tokenization), vocab size N.
  Role-equivalent to the "SP8192" upgrade recommended by the challenge.
"""

import argparse
import json
import os
import zipfile

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

TRAIN_END = 90_000_000
VALID_END = 95_000_000
EVAL_BYTES = 2_000_000


def load_enwik8(zip_path, out_dir):
    raw_path = os.path.join(out_dir, "enwik8")
    if not os.path.exists(raw_path):
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            member = [n for n in names if n.lower().endswith("enwik8")][0]
            with zf.open(member) as src, open(raw_path, "wb") as dst:
                dst.write(src.read())
    with open(raw_path, "rb") as f:
        data = f.read()
    return data


def train_tokenizer(train_text, vocab_size, tok_path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator([train_text], trainer=trainer)
    tok.save(tok_path)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--tok-sample-bytes", type=int, default=20_000_000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_enwik8(args.zip, args.out)
    assert len(data) >= VALID_END, len(data)

    train_bytes = data[:TRAIN_END]
    eval_bytes = data[TRAIN_END:TRAIN_END + EVAL_BYTES]

    tok_dir = os.path.join(args.out, "tok%d" % args.vocab)
    os.makedirs(tok_dir, exist_ok=True)
    tok_path = os.path.join(tok_dir, "bytebpe.json")

    if not os.path.exists(tok_path):
        sample = train_bytes[: args.tok_sample_bytes].decode("utf-8", errors="ignore")
        train_tokenizer(sample, args.vocab, tok_path)

    tok = Tokenizer.from_file(tok_path)

    train_ids_path = os.path.join(tok_dir, "train_ids.npy")
    eval_ids_path = os.path.join(tok_dir, "eval_ids.npy")

    if not os.path.exists(train_ids_path):
        enc = tok.encode_batch(
            [train_bytes[i:i + 5_000_000].decode("utf-8", errors="ignore")
             for i in range(0, len(train_bytes), 5_000_000)],
            add_special_tokens=False,
        )
        ids = np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in enc])
        np.save(train_ids_path, ids)

    if not os.path.exists(eval_ids_path):
        e = tok.encode(eval_bytes.decode("utf-8", errors="ignore"), add_special_tokens=False)
        np.save(eval_ids_path, np.asarray(e.ids, dtype=np.uint16))

    train_ids = np.load(train_ids_path, mmap_mode="r")
    eval_ids = np.load(eval_ids_path)
    meta = {
        "corpus": "enwik8",
        "vocab_size": args.vocab,
        "train_bytes": len(train_bytes),
        "eval_bytes": len(eval_bytes),
        "train_tokens": int(train_ids.shape[0]),
        "eval_tokens": int(eval_ids.shape[0]),
        "bytes_per_token_train": round(len(train_bytes) / float(train_ids.shape[0]), 4),
        "tokenizer": "byte-level BPE (GPT-2 style pre-tokenization)",
        "note": "NOT FineWeb; not comparable to the official leaderboard BPB.",
    }
    with open(os.path.join(tok_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("PREP_OK vocab=%d train_tokens=%d eval_tokens=%d bpt=%.3f"
          % (args.vocab, meta["train_tokens"], meta["eval_tokens"],
             meta["bytes_per_token_train"]))


if __name__ == "__main__":
    main()
