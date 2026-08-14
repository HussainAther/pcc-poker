from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze_file
from .behavioral_experiment import (
    write_adaptive_family_validation,
    write_behavioral_validation,
    write_opponent_adaptation_confirmation,
    write_predictive_control_confirmation,
)
from .counterfactual_control import write_counterfactual_control_validation
from .effective_chaos_validation import write_effective_chaos_validation
from .control_mechanism import write_control_pressure_mechanism
from .mixed import analyze_mixed_file, write_mixed_grid
from .play import play_session, write_session
from .pressure_decomposition import write_pressure_decomposition
from .policies import PURE_MIXTURES
from .robustness import write_robustness_outputs
from .transfer import analyze_family_transfer_files, write_family_transfer_grid
from .temporal_control import write_temporal_control_validation
from .simulate import (
    adaptive_pairwise_sweep,
    balanced_cycle_confirmation,
    generate_family_dataset,
    generate_mixed_dataset,
    generate_recovery_dataset,
    pairwise_sweep,
    simulate_match,
    write_jsonl,
)


def effective_chaos_validation_command(args) -> int:
    report = write_effective_chaos_validation(
        args.output,
        calibration_mixtures=args.calibration_mixtures,
        calibration_hands_per_seat=args.calibration_hands_per_seat,
        evaluation_mixtures=args.evaluation_mixtures,
        evaluation_hands_per_seat=args.evaluation_hands_per_seat,
    )
    print(json.dumps({
        "effective_chaos_construct_confirmed": report["effective_chaos_construct_confirmed"],
        "prespecified_checks": report["prespecified_checks"],
        "families": {
            family: {
                "raw": result["raw_surprisal_weight_correlations"],
                "effective": result["effective_surprisal_weight_correlations"],
                "raw_margin": result["raw_discriminant_margin"],
                "effective_margin": result["effective_discriminant_margin"],
                "shuffled_chaos": result["shuffled_chaos_weight_correlation"],
            }
            for family, result in report["families"].items()
        },
    }, indent=2))
    return 0


def pressure_decomposition_command(args) -> int:
    report = write_pressure_decomposition(
        args.output,
        replicates=args.replicates,
        calibration_hands_per_seat=args.calibration_hands_per_seat,
        evaluation_hands_per_seat=args.evaluation_hands_per_seat,
        calibration_seed=args.calibration_seed,
        evaluation_seed=args.evaluation_seed,
        seed_stride=args.seed_stride,
        purity=args.purity,
        temperature=args.temperature,
        minimum_attenuation=args.minimum_attenuation,
    )
    print(json.dumps(report, indent=2))
    return 0


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


def play_command(args) -> int:
    records, summary = play_session(
        hands=args.hands,
        opponent_mode=args.opponent,
        seed=args.seed,
        auto_human=args.auto_human,
    )
    if args.output:
        write_session(args.output, records, summary)
        print(f"Anonymous local session log: {args.output}")
    return 0


def adaptive_validation_command(args) -> int:
    report = write_adaptive_family_validation(
        args.output,
        calibration_mixtures=args.calibration_mixtures,
        calibration_hands_per_seat=args.calibration_hands_per_seat,
        evaluation_mixtures=args.evaluation_mixtures,
        evaluation_hands_per_seat=args.evaluation_hands_per_seat,
    )
    print(json.dumps({
        "matching_axis_correlations": report["families"]["adaptive"][
            "matching_axis_correlations"
        ],
        "construct_result": report["adaptive_family_construct_result"],
        "warning": report["design"]["circularity_warning"],
    }, indent=2))
    return 0


def adaptive_sweep_command(args) -> int:
    report = adaptive_pairwise_sweep(args.hands_per_seat_order, args.seed)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def balanced_cycle_command(args) -> int:
    report = balanced_cycle_confirmation(
        replicates=args.replicates,
        hands_per_seat_order=args.hands_per_seat_order,
        seed=args.seed,
        seed_stride=args.seed_stride,
        maximum_edge_ratio=args.maximum_edge_ratio,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "edges": report["edges"],
        "edge_strength_ratio": report["edge_strength_ratio"],
        "balanced_cycle_confirmed": report["balanced_cycle_confirmed"],
        "warning": report["warning"],
    }, indent=2))
    return 0


def robustness_grid_command(args) -> int:
    report = write_robustness_outputs(
        args.output,
        args.csv_output,
        temperatures=tuple(args.temperatures),
        purities=tuple(args.purities),
        hand_counts=tuple(args.hand_counts),
        replicates=args.replicates,
        seed=args.seed,
        seed_stride=args.seed_stride,
        minimum_cycle_fraction=args.minimum_cycle_fraction,
        maximum_dominance_fraction=args.maximum_dominance_fraction,
        workers=args.workers,
    )
    print(json.dumps({
        "design": report["design"],
        "aggregate": report["aggregate"],
        "json_output": args.output,
        "csv_output": args.csv_output,
        "warning": report["warning"],
    }, indent=2))
    return 0


def temporal_control_command(args) -> int:
    report = write_temporal_control_validation(
        args.output,
        training_mixtures=args.training_mixtures,
        evaluation_mixtures=args.evaluation_mixtures,
        hands_per_seat=args.hands_per_seat,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        shuffle_repetitions=args.shuffle_repetitions,
    )
    print(json.dumps({
        "static_model": report["static_model"],
        "temporal_model": report["temporal_model"],
        "shuffled_history_baseline": report["shuffled_history_baseline"],
        "trajectory_control_score": report["trajectory_control_score"],
        "prespecified_checks": report["prespecified_checks"],
        "temporal_control_confirmed": report["temporal_control_confirmed"],
        "warning": report["warning"],
    }, indent=2))
    return 0


def counterfactual_control_command(args) -> int:
    report = write_counterfactual_control_validation(
        args.output,
        replicates=args.replicates,
        calibration_hands_per_seat=args.calibration_hands_per_seat,
        evaluation_hands_per_seat=args.evaluation_hands_per_seat,
        calibration_seed=args.calibration_seed,
        evaluation_seed=args.evaluation_seed,
        seed_stride=args.seed_stride,
        purity=args.purity,
        temperature=args.temperature,
    )
    print(json.dumps({
        "mode_summary": report["mode_summary"],
        "control_specificity": report["control_specificity"],
        "prespecified_checks": report["prespecified_checks"],
        "counterfactual_control_confirmed": report[
            "counterfactual_control_confirmed"
        ],
        "warning": report["warning"],
    }, indent=2))
    return 0


def control_pressure_mechanism_command(args) -> int:
    report = write_control_pressure_mechanism(
        args.output,
        replicates=args.replicates,
        calibration_hands_per_seat=args.calibration_hands_per_seat,
        evaluation_hands_per_seat=args.evaluation_hands_per_seat,
        calibration_seed=args.calibration_seed,
        evaluation_seed=args.evaluation_seed,
        seed_stride=args.seed_stride,
        purities=tuple(args.purities),
        temperatures=tuple(args.temperatures),
    )
    print(json.dumps({
        "target_summary": report["target_summary"],
        "pressure_minus_chaos_specificity": report[
            "pressure_minus_chaos_specificity"
        ],
        "jointly_positive_pressure_cells": report[
            "jointly_positive_pressure_cells"
        ],
        "prespecified_checks": report["prespecified_checks"],
        "control_pressure_mechanism_confirmed": report[
            "control_pressure_mechanism_confirmed"
        ],
        "warning": report["warning"],
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
    family_dataset=commands.add_parser("family-dataset", help="generate data from one policy family");family_dataset.add_argument("--family",choices=["score","independent","adaptive"],required=True);family_dataset.add_argument("--mixtures",type=int,default=60);family_dataset.add_argument("--hands-per-seat",type=int,default=100);family_dataset.add_argument("--alpha",type=float,default=0.7);family_dataset.add_argument("--temperature",type=float,default=0.35);family_dataset.add_argument("--seed",type=int,required=True);family_dataset.add_argument("--output",required=True);family_dataset.set_defaults(func=family_dataset_command)
    family_transfer=commands.add_parser("family-transfer", help="train on one policy family and test another");family_transfer.add_argument("--training",required=True);family_transfer.add_argument("--transfer",required=True);family_transfer.add_argument("--output",required=True);family_transfer.set_defaults(func=family_transfer_command)
    transfer_grid=commands.add_parser("family-transfer-grid", help="replicate bidirectional family transfer");transfer_grid.add_argument("--score-seeds",type=int,nargs="+",default=[61,62,63,64,65]);transfer_grid.add_argument("--independent-seeds",type=int,nargs="+",default=[71,72,73,74,75]);transfer_grid.add_argument("--mixtures",type=int,default=40);transfer_grid.add_argument("--hands-per-seat",type=int,default=75);transfer_grid.add_argument("--alpha",type=float,default=0.7);transfer_grid.add_argument("--shuffle-repetitions",type=int,default=10);transfer_grid.add_argument("--output",required=True);transfer_grid.set_defaults(func=family_transfer_grid_command)
    behavioral=commands.add_parser("behavioral-validation", help="validate label-free behavioral PCC measures");behavioral.add_argument("--calibration-mixtures",type=int,default=20);behavioral.add_argument("--calibration-hands-per-seat",type=int,default=25);behavioral.add_argument("--evaluation-mixtures",type=int,default=30);behavioral.add_argument("--evaluation-hands-per-seat",type=int,default=50);behavioral.add_argument("--output",required=True);behavioral.set_defaults(func=behavioral_validation_command)
    control=commands.add_parser("control-confirmation", help="confirm opponent-adaptive Control on fresh seeds");control.add_argument("--calibration-mixtures",type=int,default=20);control.add_argument("--calibration-hands-per-seat",type=int,default=25);control.add_argument("--evaluation-mixtures",type=int,default=60);control.add_argument("--evaluation-hands-per-seat",type=int,default=100);control.add_argument("--output",required=True);control.set_defaults(func=control_confirmation_command)
    play=commands.add_parser("play", help="play heads-up Leduc against an Adaptive PCC opponent");play.add_argument("--opponent",choices=["pressure","control","chaos"],default="control");play.add_argument("--hands",type=int,default=6);play.add_argument("--seed",type=int,default=701);play.add_argument("--output");play.add_argument("--auto-human",action="store_true",help=argparse.SUPPRESS);play.set_defaults(func=play_command)
    adaptive=commands.add_parser("adaptive-validation", help="validate the playable Adaptive PCC family");adaptive.add_argument("--calibration-mixtures",type=int,default=20);adaptive.add_argument("--calibration-hands-per-seat",type=int,default=25);adaptive.add_argument("--evaluation-mixtures",type=int,default=60);adaptive.add_argument("--evaluation-hands-per-seat",type=int,default=100);adaptive.add_argument("--output",required=True);adaptive.set_defaults(func=adaptive_validation_command)
    adaptive_sweep=commands.add_parser("adaptive-sweep", help="diagnose balance among playable PCC opponents");adaptive_sweep.add_argument("--hands-per-seat-order",type=int,default=4000);adaptive_sweep.add_argument("--seed",type=int,default=901);adaptive_sweep.add_argument("--output",required=True);adaptive_sweep.set_defaults(func=adaptive_sweep_command)
    balanced_cycle=commands.add_parser("balanced-cycle", help="confirm the frozen engineered PCC cycle across fresh replicates");balanced_cycle.add_argument("--replicates",type=int,default=12);balanced_cycle.add_argument("--hands-per-seat-order",type=int,default=1000);balanced_cycle.add_argument("--seed",type=int,default=23001);balanced_cycle.add_argument("--seed-stride",type=int,default=20);balanced_cycle.add_argument("--maximum-edge-ratio",type=float,default=3.0);balanced_cycle.add_argument("--output",required=True);balanced_cycle.set_defaults(func=balanced_cycle_command)
    robustness=commands.add_parser("robustness-grid", help="test the frozen v0.3 cycle across temperature, purity, and match length");robustness.add_argument("--temperatures",type=float,nargs="+",default=[0.20,0.35,0.50,0.75]);robustness.add_argument("--purities",type=float,nargs="+",default=[0.70,0.80,0.90,1.00]);robustness.add_argument("--hand-counts",type=int,nargs="+",default=[250,1000,4000]);robustness.add_argument("--replicates",type=int,default=10);robustness.add_argument("--seed",type=int,default=41001);robustness.add_argument("--seed-stride",type=int,default=20);robustness.add_argument("--minimum-cycle-fraction",type=float,default=0.80);robustness.add_argument("--maximum-dominance-fraction",type=float,default=0.20);robustness.add_argument("--workers",type=int);robustness.add_argument("--output",required=True);robustness.add_argument("--csv-output",required=True);robustness.set_defaults(func=robustness_grid_command)
    temporal_control=commands.add_parser("temporal-control-validation", help="test whether prior opponent history improves held-out Control detection");temporal_control.add_argument("--training-mixtures",type=int,default=80);temporal_control.add_argument("--evaluation-mixtures",type=int,default=80);temporal_control.add_argument("--hands-per-seat",type=int,default=100);temporal_control.add_argument("--training-seed",type=int,default=61001);temporal_control.add_argument("--evaluation-seed",type=int,default=62001);temporal_control.add_argument("--shuffle-repetitions",type=int,default=25);temporal_control.add_argument("--output",required=True);temporal_control.set_defaults(func=temporal_control_command)
    counterfactual=commands.add_parser("counterfactual-control", help="intervene on opponent-model alignment under frozen policies");counterfactual.add_argument("--replicates",type=int,default=16);counterfactual.add_argument("--calibration-hands-per-seat",type=int,default=250);counterfactual.add_argument("--evaluation-hands-per-seat",type=int,default=500);counterfactual.add_argument("--calibration-seed",type=int,default=71001);counterfactual.add_argument("--evaluation-seed",type=int,default=81001);counterfactual.add_argument("--seed-stride",type=int,default=1000);counterfactual.add_argument("--purity",type=float,default=0.8);counterfactual.add_argument("--temperature",type=float,default=0.35);counterfactual.add_argument("--output",required=True);counterfactual.set_defaults(func=counterfactual_control_command)
    mechanism=commands.add_parser("control-pressure-mechanism", help="test contextual prediction in the Control-over-Pressure edge");mechanism.add_argument("--replicates",type=int,default=16);mechanism.add_argument("--calibration-hands-per-seat",type=int,default=250);mechanism.add_argument("--evaluation-hands-per-seat",type=int,default=500);mechanism.add_argument("--calibration-seed",type=int,default=91001);mechanism.add_argument("--evaluation-seed",type=int,default=101001);mechanism.add_argument("--seed-stride",type=int,default=2000);mechanism.add_argument("--purities",type=float,nargs="+",default=[0.70,0.80,0.90]);mechanism.add_argument("--temperatures",type=float,nargs="+",default=[0.25,0.35,0.50]);mechanism.add_argument("--output",required=True);mechanism.set_defaults(func=control_pressure_mechanism_command)
    effective_chaos=commands.add_parser("effective-chaos-validation", help="validate the independent value-floor Chaos candidate on fresh synthetic mixtures");effective_chaos.add_argument("--calibration-mixtures",type=int,default=20);effective_chaos.add_argument("--calibration-hands-per-seat",type=int,default=25);effective_chaos.add_argument("--evaluation-mixtures",type=int,default=60);effective_chaos.add_argument("--evaluation-hands-per-seat",type=int,default=100);effective_chaos.add_argument("--output",required=True);effective_chaos.set_defaults(func=effective_chaos_validation_command)
    decomposition=commands.add_parser("pressure-decomposition", help="decompose which engineered Pressure term sustains Control contextual alignment");decomposition.add_argument("--replicates",type=int,default=16);decomposition.add_argument("--calibration-hands-per-seat",type=int,default=250);decomposition.add_argument("--evaluation-hands-per-seat",type=int,default=500);decomposition.add_argument("--calibration-seed",type=int,default=111001);decomposition.add_argument("--evaluation-seed",type=int,default=121001);decomposition.add_argument("--seed-stride",type=int,default=2000);decomposition.add_argument("--purity",type=float,default=0.8);decomposition.add_argument("--temperature",type=float,default=0.35);decomposition.add_argument("--minimum-attenuation",type=float,default=0.50);decomposition.add_argument("--output",required=True);decomposition.set_defaults(func=pressure_decomposition_command)
    return root


def main() -> int:
    args=parser().parse_args();return args.func(args)
