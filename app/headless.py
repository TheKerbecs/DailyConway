"""
Headless miner: runs GPUMiner with custom rules and saves every hit as a JSON file.

Usage examples:
    uv run python -m app.headless --config headless.json
    uv run python -m app.headless \
        --output-dir ./hits \
        --blocks 2048 --threads 256 --ipt 256 --workers 4 \
        --group front:513 \
        --group any:b0b1400,any:626f62696e6f75

    uv run python -m app.headless --output-dir ./hits \
        --regex '^513.*(b0b1400|626f62696e6f75)' --regex-anchor b0b1400

The config JSON, when provided, has the same shape as the WebSocket "start" payload:
    {
      "blocks": 2048, "threads": 256, "ipt": 256, "workers": 4,
      "owner": "Bobinou",
      "output_dir": "./hits",
      "groups": [
        [{"position": "front", "value": "513"}],
        [{"position": "any", "value": "b0b1400"},
         {"position": "any", "value": "626f62696e6f75"}]
      ]
      // OR, instead of "groups":
      // "regex": {"pattern": "^513.*(b0b1400|626f62696e6f75)", "flags": "", "anchor": "b0b1400"}
    }
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

# Ensure NIST-free custom mode and the same DLL/so discovery as website mode.
from .miner import GPUMiner


def _bin_to_rle(b_str: str) -> str:
    """Convert a 256-char binary string into a Life RLE block."""
    if len(b_str) != 256:
        return ""
    parts = []
    for r in range(16):
        row = b_str[r * 16:(r + 1) * 16]
        rle_row = ""
        count = 0
        cur = row[0]
        for ch in row:
            if ch == cur:
                count += 1
            else:
                tag = "b" if cur == "0" else "o"
                rle_row += (str(count) if count > 1 else "") + tag
                cur = ch
                count = 1
        if cur == "1":
            rle_row += (str(count) if count > 1 else "") + "o"
        parts.append(rle_row)
    body = "$".join(parts) + "!"
    chunks = [body[i:i + 70] for i in range(0, len(body), 70)]
    return "x = 16, y = 16, rule = B3/S23:T16,16\n" + "\n".join(chunks)


def _parse_group_arg(s: str) -> list[dict]:
    """Parse a comma-separated list of position:value entries into a group."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Group entry {part!r} must be position:value (front|back|any:hex).")
        pos, val = part.split(":", 1)
        pos = pos.strip().lower()
        val = val.strip().lower()
        if pos not in ("front", "back", "any"):
            raise ValueError(f"Invalid position {pos!r} in {part!r}.")
        if not val or len(val) > 16 or any(c not in "0123456789abcdef" for c in val):
            raise ValueError(f"Invalid hex value {val!r} in {part!r}.")
        out.append({"position": pos, "value": val})
    if not out:
        raise ValueError(f"Empty group spec: {s!r}")
    return out


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="app.headless", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Path to JSON config (overrides flags when keys overlap).")
    p.add_argument("--output-dir", default="./hits", help="Directory to write hit JSON files (default: ./hits).")
    p.add_argument("--owner", default=os.environ.get("MINER_OWNER", "Headless"), help="Owner string embedded in payloads.")
    p.add_argument("--blocks", type=int, default=2048)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--ipt", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--group", action="append", default=[],
                   help="An AND-group as 'pos:val[,pos:val,...]'. Repeat for multiple AND groups.")
    p.add_argument("--regex", help="Python regex pattern. Mutually exclusive with --group.")
    p.add_argument("--regex-flags", default="", help="Subset of i s m x.")
    p.add_argument("--regex-anchor", default="", help="GPU pre-filter anchor (1-16 lowercase hex). Required with --regex.")
    p.add_argument("--no-rle", action="store_true", help="Don't include the RLE in saved files.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-tick log output.")
    return p.parse_args()


def _resolve_config(args: argparse.Namespace) -> dict:
    """Merge CLI flags with optional JSON config (config takes precedence)."""
    cfg: dict = {
        "output_dir": args.output_dir,
        "owner": args.owner,
        "blocks": args.blocks,
        "threads": args.threads,
        "ipt": args.ipt,
        "workers": args.workers,
        "no_rle": args.no_rle,
        "quiet": args.quiet,
    }

    if args.regex:
        if args.group:
            raise SystemExit("--regex and --group are mutually exclusive.")
        cfg["regex"] = {
            "pattern": args.regex,
            "flags": args.regex_flags,
            "anchor": args.regex_anchor,
        }
    elif args.group:
        cfg["groups"] = [_parse_group_arg(g) for g in args.group]

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        cfg.update(file_cfg)  # file values win

    if "groups" not in cfg and "regex" not in cfg:
        raise SystemExit("Provide --group ... or --regex ... or specify groups/regex in --config.")
    if "groups" in cfg and "regex" in cfg and cfg["regex"]:
        raise SystemExit("Config has both 'groups' and 'regex'; pick one.")

    return cfg


class _HitWriter:
    """Writes each hit as a JSON file into the configured output directory."""

    def __init__(self, output_dir: str, include_rle: bool, quiet: bool) -> None:
        self.dir = Path(output_dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.include_rle = include_rle
        self.quiet = quiet
        self.count = 0
        self.start_ts = time.time()

    async def on_log(self, msg: str) -> None:
        if not self.quiet:
            print(f"[log] {msg}", flush=True)

    async def on_rate(self, rate: float, hits_rate: float, queue: int, dropped: int) -> None:
        if self.quiet:
            return
        elapsed = time.time() - self.start_ts
        print(f"[rate] {rate/1e6:6.1f} M/s  hits/s={hits_rate:5.1f}  q={queue:4d}  drop={dropped}  saved={self.count}  uptime={int(elapsed)}s",
              flush=True)

    async def on_hit_count(self, total: int) -> None:
        # Logged via on_rate already; nothing to do here.
        return

    async def _write_one(self, mid: str, gpu_hex: str, iters: int, peak: int, payload: dict) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        # File name: {timestamp}_{iters}_{first16ofhash}.json — sortable by time then quality.
        fname = f"{ts}_{iters:05d}_{gpu_hex[:16]}_{mid[:8]}.json"
        path = self.dir / fname
        record = dict(payload)
        record["match_id"] = mid
        record["saved_at"] = int(time.time() * 1000)
        if self.include_rle:
            b = record.get("bin")
            if isinstance(b, str):
                record["rle"] = _bin_to_rle(b)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, path)
        self.count += 1
        if not self.quiet:
            print(f"[hit] iters={iters:>5} peak={peak:>4} -> {path.name}", flush=True)

    async def on_match_history(self, mid: str, gpu_hex: str, iters: int, peak: int, payload: dict) -> None:
        await self._write_one(mid, gpu_hex, iters, peak, payload)

    async def on_match_history_batch(self, items) -> None:
        for mid, gpu_hex, iters, peak, payload in items:
            await self._write_one(mid, gpu_hex, iters, peak, payload)

    async def on_stop(self) -> None:
        if not self.quiet:
            print(f"[stop] total saved: {self.count}", flush=True)


async def _amain(cfg: dict) -> None:
    os.environ.setdefault("MINER_OWNER", str(cfg.get("owner", "Headless")))

    writer = _HitWriter(cfg["output_dir"], include_rle=not cfg.get("no_rle", False), quiet=bool(cfg.get("quiet", False)))
    miner = GPUMiner(callbacks={
        "on_log": writer.on_log,
        "on_rate": writer.on_rate,
        "on_hit_count": writer.on_hit_count,
        "on_match_history": writer.on_match_history,
        "on_match_history_batch": writer.on_match_history_batch,
        "on_stop": writer.on_stop,
    })

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop(*_):
        if not stop_event.is_set():
            print("[signal] stop requested, draining...", flush=True)
            stop_event.set()
            miner.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _request_stop())

    await miner.run(
        target_suite="any",
        blocks=int(cfg["blocks"]),
        threads=int(cfg["threads"]),
        ipt=int(cfg["ipt"]),
        n_workers=int(cfg["workers"]),
        custom_mode=True,
        groups=cfg.get("groups"),
        regex=cfg.get("regex"),
    )


def main() -> None:
    args = _build_args()
    cfg = _resolve_config(args)
    print(f"[headless] output_dir={cfg['output_dir']}", flush=True)
    if "regex" in cfg and cfg["regex"]:
        print(f"[headless] regex=/{cfg['regex'].get('pattern','')}/{cfg['regex'].get('flags','')} "
              f"anchor={cfg['regex'].get('anchor','')}", flush=True)
    else:
        print(f"[headless] groups={json.dumps(cfg.get('groups'))}", flush=True)
    try:
        asyncio.run(_amain(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
