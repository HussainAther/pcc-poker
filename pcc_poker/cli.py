from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze_file
from .behavioral_experiment import (
    write_behavioral_validation,
    write_opponent_adaptation_confirmation,
    write_predictive_control_confirmation,
)
from .mixed import analyze_mixed_file, write_mixed_grid
from .policies import PURE_MIXTURES
from .transfer import analyze_family_transfer_files, write_family_transfer_grid
from .simulate import (
    generate_family_dataset,
    generate_mixed_dataset,
    generate_recovery_dataset,
    pairwise_sweep,
    simulate_match,
    write_jsonl,
)


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


def mixed_dataset_command(args) -> int:
    records, summary = generate_mixed_dataset(
        mixtures=args.mixtures,
        hands_per_seat=args.hands_per_seat,
        seed=args.seed,
        alpha=args.alpha,
        focal_temperature=args.temperature,
    )
    write_jsonl(args.output, records)
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}, indent=2))
    return 0


def analyze_command(args) -> int:
    print(json.dumps(analyze_file(args.input,args.output),indent=2));return 0


def mixed_analyze_command(args) -> int:
    print(json.dumps(analyze_mixed_file(args.input, args.output), indent=2))
    return 0


def mixed_grid_command(args) -> int:
    report = write_mixed_grid(
        args.output,
        seeds=tuple(args.seeds),
        temperatures=tuple(args.temperatures),
        mixtures=args.mixtures,
        hands_per_seat=args.hands_per_seat,
        alpha=args.alpha,
        shuffle_repetitions=args.shuffle_repetitions,
    )
    print(json.dumps(report["aggregate"], indent=2))
    return 0


def family_dataset_command(args) -> int:
    records, summary = generate_family_dataset(
        family=args.family,
        mixtures=args.mixtures,
        hands_per_seat=args.hands_per_seat,
        seed=args.seed,
        alpha=args.alpha,
        focal_temperature=args.temperature,
    )
    write_jsonl(args.output, records)
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}, indent=2))
    return 0


def family_transfer_command(args) -> int:
    report = analyze_family_transfer_files(
        args.training, args.transfer, args.output
    )
    print(json.dumps(report, indent=2))
    return 0


def family_transfer_grid_command(args) -> int:
    seed_pairs = tuple(zip(args.score_seeds, args.independent_seeds))
    if len(args.score_seeds) != len(args.independent_seeds):
        raise ValueError("score and independent seed lists must have equal length")
    report = write_family_transfer_grid(
        args.output,
        seed_pairs=seed_pairs,
        mixtures=args.mixtures,
        hands_per_seat=args.hands_per_seat,
        alpha=args.alpha,
        shuffle_repetitions=args.shuffle_repetitions,
    )
    print(json.dumps(report["aggregate_by_direction"], indent=2))
    return 0


def behavioral_validation_command(args) -> int:
    report = write_behavioral_validation(
        args.output,
        calibration_mixtures=args.calibration_mixtures,
        calibration_hands_per_seat=args.calibration_hands_per_seat,
        evaluation_mixtures=args.evaluation_mixtures,
        evaluation_hands_per_seat=args.evaluation_hands_per_seat,
    )
    print(json.dumps({
        "families": report["families"],
        "matching_axis_positive_in_every_family": report["matching_axis_positive_in_every_family"],
    }, indent=2))
    return 0


def control_confirmation_command(args) -> int:
    report = write_opponent_adaptation_confirmation(
        args.output,
        calibration_mixtures=args.calibration_mixtures,
        calibration_hands_per_seat=args.calibration_hands_per_seat,
        evaluation_mixtures=args.evaluation_mixtures,
        evaluation_hands_per_seat=args.evaluation_hands_per_seat,
    )
    print(json.dumps({
        "matching_axis_correlations": {
            family: result["matching_axis_correlations"]
            for family, result in report["families"].items()
        },
        "cross_family_construct_result": report["cross_family_construct_result"],
    }, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root=argparse.ArgumentParser(description="PCC experiments in Leduc poker");commands=root.add_subparsers(required=True)
    sim=commands.add_parser("simulate");sim.add_argument("--hands",type=int,default=3000);sim.add_argument("--mode0",choices=PURE_MIXTURES,default="pressure");sim.add_argument("--mode1",choices=PURE_MIXTURES,default="control");sim.add_argument("--seed",type=int,default=7);sim.add_argument("--output",required=True);sim.set_defaults(func=simulate_command)
    sweep=commands.add_parser("sweep");sweep.add_argument("--hands-per-matchup",type=int,default=2000);sweep.add_argument("--seed",type=int,default=17);sweep.add_argument("--output",required=True);sweep.set_defaults(func=sweep_command)
    dataset=commands.add_parser("dataset");dataset.add_argument("--hands-per-seat",type=int,default=500);dataset.add_argument("--seed",type=int,default=23);dataset.add_argument("--output",required=True);dataset.set_defaults(func=dataset_command)
    mixed_dataset=commands.add_parser("mixed-dataset", help="generate continuous PCC mixtures");mixed_dataset.add_argument("--mixtures",type=int,default=60);mixed_dataset.add_argument("--hands-per-seat",type=int,default=100);mixed_dataset.add_argument("--alpha",type=float,default=0.7);mixed_dataset.add_argument("--temperature",type=float,default=0.35);mixed_dataset.add_argument("--seed",type=int,default=41);mixed_dataset.add_argument("--output",required=True);mixed_dataset.set_defaults(func=mixed_dataset_command)
    analyze=commands.add_parser("analyze");analyze.add_argument("input");analyze.add_argument("--output",required=True);analyze.set_defaults(func=analyze_command)
    mixed_analyze=commands.add_parser("mixed-analyze", help="recover held-out mixture weights");mixed_analyze.add_argument("input");mixed_analyze.add_argument("--output",required=True);mixed_analyze.set_defaults(func=mixed_analyze_command)
    mixed_grid=commands.add_parser("mixed-grid", help="replicate recovery over seeds and temperatures");mixed_grid.add_argument("--seeds",type=int,nargs="+",default=[41,42,43,44,45]);mixed_grid.add_argument("--temperatures",type=float,nargs="+",default=[0.25,0.35,0.5]);mixed_grid.add_argument("--mixtures",type=int,default=60);mixed_grid.add_argument("--hands-per-seat",type=int,default=100);mixed_grid.add_argument("--alpha",type=float,default=0.7);mixed_grid.add_argument("--shuffle-repetitions",type=int,default=25);mixed_grid.add_argument("--output",required=True);mixed_grid.set_defaults(func=mixed_grid_command)
    family_dataset=commands.add_parser("family-dataset", help="generate data from one policy family");family_dataset.add_argument("--family",choices=["score","independent"],required=True);family_dataset.add_argument("--mixtures",type=int,default=60);family_dataset.add_argument("--hands-per-seat",type=int,default=100);family_dataset.add_argument("--alpha",type=float,default=0.7);family_dataset.add_argument("--temperature",type=float,default=0.35);family_dataset.add_argument("--seed",type=int,required=True);family_dataset.add_argument("--output",required=True);family_dataset.set_defaults(func=family_dataset_command)
    family_transfer=commands.add_parser("family-transfer", help="train on one policy family and test another");family_transfer.add_argument("--training",required=True);family_transfer.add_argument("--transfer",required=True);family_transfer.add_argument("--output",required=True);family_transfer.set_defaults(func=family_transfer_command)
    transfer_grid=commands.add_parser("family-transfer-grid", help="replicate bidirectional family transfer");transfer_grid.add_argument("--score-seeds",type=int,nargs="+",default=[61,62,63,64,65]);transfer_grid.add_argument("--independent-seeds",type=int,nargs="+",default=[71,72,73,74,75]);transfer_grid.add_argument("--mixtures",type=int,default=40);transfer_grid.add_argument("--hands-per-seat",type=int,default=75);transfer_grid.add_argument("--alpha",type=float,default=0.7);transfer_grid.add_argument("--shuffle-repetitions",type=int,default=10);transfer_grid.add_argument("--output",required=True);transfer_grid.set_defaults(func=family_transfer_grid_command)
    behavioral=commands.add_parser("behavioral-validation", help="validate label-free behavioral PCC measures");behavioral.add_argument("--calibration-mixtures",type=int,default=20);behavioral.add_argument("--calibration-hands-per-seat",type=int,default=25);behavioral.add_argument("--evaluation-mixtures",type=int,default=30);behavioral.add_argument("--evaluation-hands-per-seat",type=int,default=50);behavioral.add_argument("--output",required=True);behavioral.set_defaults(func=behavioral_validation_command)
    control=commands.add_parser("control-confirmation", help="confirm opponent-adaptive Control on fresh seeds");control.add_argument("--calibration-mixtures",type=int,default=20);control.add_argument("--calibration-hands-per-seat",type=int,default=25);control.add_argument("--evaluation-mixtures",type=int,default=60);control.add_argument("--evaluation-hands-per-seat",type=int,default=100);control.add_argument("--output",required=True);control.set_defaults(func=control_confirmation_command)
    return root


def main() -> int:
    args=parser().parse_args();return args.func(args)
