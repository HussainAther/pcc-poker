from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze_file
from .policies import PURE_MIXTURES
from .simulate import generate_recovery_dataset, pairwise_sweep, simulate_match, write_jsonl


def simulate_command(args) -> int:
    records, summary = simulate_match(args.hands, PURE_MIXTURES[args.mode0], PURE_MIXTURES[args.mode1], args.seed, args.mode0, args.mode1)
    write_jsonl(args.output, records)
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2));return 0


def sweep_command(args) -> int:
    report=pairwise_sweep(args.hands_per_matchup,args.seed);target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2));return 0


def dataset_command(args) -> int:
    records, summary = generate_recovery_dataset(args.hands_per_seat, args.seed)
    write_jsonl(args.output, records)
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)); return 0


def analyze_command(args) -> int:
    print(json.dumps(analyze_file(args.input,args.output),indent=2));return 0


def parser() -> argparse.ArgumentParser:
    root=argparse.ArgumentParser(description="PCC experiments in Leduc poker");commands=root.add_subparsers(required=True)
    sim=commands.add_parser("simulate");sim.add_argument("--hands",type=int,default=3000);sim.add_argument("--mode0",choices=PURE_MIXTURES,default="pressure");sim.add_argument("--mode1",choices=PURE_MIXTURES,default="control");sim.add_argument("--seed",type=int,default=7);sim.add_argument("--output",required=True);sim.set_defaults(func=simulate_command)
    sweep=commands.add_parser("sweep");sweep.add_argument("--hands-per-matchup",type=int,default=2000);sweep.add_argument("--seed",type=int,default=17);sweep.add_argument("--output",required=True);sweep.set_defaults(func=sweep_command)
    dataset=commands.add_parser("dataset");dataset.add_argument("--hands-per-seat",type=int,default=500);dataset.add_argument("--seed",type=int,default=23);dataset.add_argument("--output",required=True);dataset.set_defaults(func=dataset_command)
    analyze=commands.add_parser("analyze");analyze.add_argument("input");analyze.add_argument("--output",required=True);analyze.set_defaults(func=analyze_command)
    return root


def main() -> int:
    args=parser().parse_args();return args.func(args)
