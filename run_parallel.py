#!/usr/bin/env python
"""Parallel batch runner — saturate the GPU by dubbing several videos at once.

A single Chatterbox stream only uses ~35% of an RTX 4080 (generation is
autoregressive, one cue at a time), so wall-clock is dominated by GPU idle time,
not memory. This launches N worker processes over DISJOINT videos; each worker is
the ordinary ``dub.py`` with a CUDA memory-fraction cap so N of them co-fit in
VRAM. Idempotent: finished outputs are skipped, so it also resumes cleanly.

    python run_parallel.py --config config.yaml --workers 2
    python run_parallel.py --config config.yaml --workers 3 --mem-fraction 0.3

Each worker streams its log to ``logs/parallel_worker<i>.log``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dubbing.config import Config
from dubbing.pairing import pair_inputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config shared by all workers.")
    ap.add_argument("--workers", type=int, default=2, help="number of GPU workers (default 2).")
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="VRAM cap per worker (default ~0.9/workers).")
    ap.add_argument("--only", nargs="*", help="restrict to these episode keys before sharding.")
    args = ap.parse_args()

    cfg = Config.load_yaml(args.config) if args.config else Config()
    if args.only:
        cfg = cfg.merged_with({"only": tuple(args.only)})

    pairing = pair_inputs(cfg)

    # Idempotent: only videos that don't already have a complete output.
    todo: list[str] = []
    for p in pairing.pairs:
        out = cfg.output_path / f"{p.video.stem}.mp4"
        if out.exists() and out.stat().st_size > 1024:
            continue
        todo.append(p.key)

    if not todo:
        print("Nothing to do — every paired video already has an output.")
        return 0

    workers = max(1, min(args.workers, len(todo)))
    frac = args.mem_fraction if args.mem_fraction else round(0.9 / workers, 3)
    # Round-robin shard so each worker gets a balanced mix.
    shards = [todo[i::workers] for i in range(workers) if todo[i::workers]]

    print(f"{len(todo)} video(s) to dub across {len(shards)} worker(s), "
          f"VRAM cap {frac} each:")
    for i, s in enumerate(shards):
        print(f"  worker {i}: {', '.join(s)}")

    cfg.log_path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Reduce fragmentation so capped workers coexist without OOM.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    procs = []
    for i, shard in enumerate(shards):
        cmd = [sys.executable, "dub.py", "--only", *shard,
               "--cuda-mem-fraction", str(frac)]
        if args.config:
            cmd += ["--config", args.config]
        log_path = cfg.log_path / f"parallel_worker{i}.log"
        fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((i, proc, fh, log_path))
        print(f"  -> launched worker {i} (pid {proc.pid}) -> {log_path}")

    print("\nWorkers running. Tail a worker log to watch progress, e.g.:")
    print("  Get-Content logs/parallel_worker0.log -Wait -Tail 20\n")

    rc = 0
    started = time.time()
    for i, proc, fh, log_path in procs:
        r = proc.wait()
        fh.close()
        mins = (time.time() - started) / 60
        print(f"worker {i} exited with code {r} (elapsed {mins:.1f} min)")
        rc = rc or r

    print("All workers finished." if rc == 0 else f"Done with errors (rc={rc}); see worker logs.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
