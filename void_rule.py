"""Void rule for ablation runs (C2G).

Purpose: decide which runs must be excluded from every mean / delta because the
host was saturated while they ran, without a human eyeballing tok/s.

Reference set (peers) for a run = runs with the SAME n_params whose train_secs
is within max(15s, 25%) of this run's train_secs. That keeps different budgets
(90s vs 150s) and different model sizes from contaminating each other.

Decision:
  INVALID if tokens_per_sec < 0.5 * median(tok/s of peers incl. self)
  INVALID if steps        < 0.5 * max(steps of peers incl. self)
  If no peer exists, only the tok/s test against the global median applies
  (a step count is meaningless without a comparable budget).

Usage:
  python void_rule.py --dir logs
  python void_rule.py --dir logs --json
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

SEED_SUFFIX = re.compile(r"_s\d+[a-z]?$")


def collect(target_dir):
    runs = []
    for p in glob.glob(os.path.join(target_dir, "**", "*-summary.json"), recursive=True):
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for k in ("bpb", "tag", "tokens_per_sec", "steps", "train_secs", "n_params"):
            if k not in d:
                break
        else:
            runs.append({
                "tag": str(d["tag"]),
                "group": SEED_SUFFIX.sub("", str(d["tag"])),
                "seed": d.get("seed"),
                "bpb": float(d["bpb"]),
                "steps": int(d["steps"]),
                "tokps": float(d["tokens_per_sec"]),
                "secs": float(d["train_secs"]),
                "n_params": int(d["n_params"]),
                "path": os.path.relpath(p, target_dir).replace("\\", "/"),
            })
    runs.sort(key=lambda r: (r["group"], str(r["seed"])))
    return runs


def judge(runs):
    global_tokps = statistics.median([r["tokps"] for r in runs]) if runs else 0.0

    groups = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)

    for r in runs:
        peers = [p for p in runs
                 if p["n_params"] == r["n_params"]
                 and abs(p["secs"] - r["secs"]) <= max(15.0, 0.25 * r["secs"])]
        ref_tokps = statistics.median([p["tokps"] for p in peers])
        # Steps are only comparable inside one exact config (same group): block
        # size changes steps/sec, so cross-config step comparison would void
        # healthy runs (e.g. abl_blk128 does ~2x the steps of abl_base).
        same_cfg = groups[r["group"]]
        ref_steps = max(p["steps"] for p in same_cfg) if len(same_cfg) > 1 else None

        reasons = []
        if ref_tokps > 0 and r["tokps"] < 0.5 * ref_tokps:
            reasons.append("tok/s %.0f < 50%% of peer median %.0f (n=%d)" % (r["tokps"], ref_tokps, len(peers)))
        if ref_steps is not None and r["steps"] < 0.5 * ref_steps:
            reasons.append("steps %d < 50%% of peer max %d (n=%d)" % (r["steps"], ref_steps, len(peers)))
        if not peers and global_tokps > 0 and r["tokps"] < 0.5 * global_tokps:
            reasons.append("tok/s %.0f < 50%% of global median %.0f (no peer)" % (r["tokps"], global_tokps))

        r["verdict"] = "INVALID" if reasons else "VALID"
        r["reason"] = "; ".join(reasons) if reasons else "-"
        r["peers"] = len(peers)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print("ERR no such dir: " + args.dir)
        return 2

    runs = judge(collect(args.dir))
    if not runs:
        print("ERR no summary json found under " + args.dir)
        return 2

    if args.json:
        print(json.dumps({"runs": runs}, ensure_ascii=True, indent=2))
        return 0

    print("| tag | seed | bpb | steps | train_s | tok/s | verdict | reason |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in runs:
        print("| %s | %s | %.5f | %d | %.1f | %.0f | %s | %s |" % (
            r["tag"], r["seed"], r["bpb"], r["steps"], r["secs"], r["tokps"], r["verdict"], r["reason"]))

    bad = [r for r in runs if r["verdict"] == "INVALID"]
    print("")
    print("TOTAL=%d VALID=%d INVALID=%d" % (len(runs), len(runs) - len(bad), len(bad)))
    for r in bad:
        print("VOID " + r["tag"] + " :: " + r["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
