from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    claim: str
    status: str
    evidence: str
    source: str
    scope: str


def _load(root: Path, rel: str) -> dict | None:
    path = root / rel
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_status(value: bool, checks: dict | None = None) -> str:
    if value:
        return "confirmed"
    if checks:
        values = [bool(v) for v in checks.values() if isinstance(v, bool)]
        if values and any(values):
            return "partial"
    return "failed"


def _fmt_bool_checks(checks: dict) -> str:
    passed = [k for k, v in checks.items() if v is True]
    failed = [k for k, v in checks.items() if v is False]
    pieces = []
    if passed:
        pieces.append(f"passed {len(passed)}/{len(passed)+len(failed)} prespecified checks")
    if failed:
        pieces.append("failed: " + ", ".join(failed))
    return "; ".join(pieces)


def build_research_status(root: str | Path = ".") -> dict:
    root = Path(root).resolve()
    rows: list[ClaimRow] = []
    missing: list[str] = []

    def get(rel: str):
        data = _load(root, rel)
        if data is None:
            missing.append(rel)
        return data

    balanced = get("validation/balanced-cycle.json")
    if balanced:
        rows.append(ClaimRow(
            "engineered-cycle",
            "Frozen engineered PCC policies exhibit the prespecified balanced Pressure→Chaos→Control→Pressure cycle.",
            _bool_status(bool(balanced.get("balanced_cycle_confirmed"))),
            f"balanced_cycle_confirmed={balanced.get('balanced_cycle_confirmed')}; edge_strength_ratio={balanced.get('edge_strength_ratio'):.3f}",
            "validation/balanced-cycle.json",
            "synthetic frozen policy family",
        ))

    robustness = get("validation/robustness-grid.json")
    if robustness:
        agg = robustness.get("aggregate", {})
        checks = {
            "cycle_fraction_passed": agg.get("cycle_fraction_passed"),
            "no_mode_dominates_grid": agg.get("no_mode_dominates_grid"),
        }
        rows.append(ClaimRow(
            "cycle-robustness",
            "The frozen PCC cycle is robust across the prespecified temperature, purity, and match-length grid.",
            _bool_status(bool(agg.get("robustness_confirmed")), checks),
            f"cycle_fraction={agg.get('cycle_fraction', 0):.3f}; no_mode_dominates_grid={agg.get('no_mode_dominates_grid')}",
            "validation/robustness-grid.json",
            "synthetic robustness surface",
        ))

    mech = get("validation/control-pressure-mechanism.json")
    if mech:
        checks = mech.get("prespecified_checks", {})
        rows.append(ClaimRow(
            "control-pressure-mechanism",
            "Contextually aligned Control specifically improves performance against Pressure under matched counterfactual controls.",
            _bool_status(bool(mech.get("control_pressure_mechanism_confirmed")), checks),
            _fmt_bool_checks(checks),
            "validation/control-pressure-mechanism.json",
            "synthetic causal-mechanism intervention",
        ))

    counter = get("validation/counterfactual-control.json")
    if counter:
        checks = counter.get("prespecified_checks", {})
        rows.append(ClaimRow(
            "generic-counterfactual-control",
            "Generic model-alignment counterfactuals identify a Control-specific dependence across targets.",
            _bool_status(bool(counter.get("counterfactual_control_confirmed")), checks),
            _fmt_bool_checks(checks),
            "validation/counterfactual-control.json",
            "synthetic counterfactual intervention",
        ))

    temporal = get("validation/temporal-control.json")
    if temporal:
        checks = temporal.get("prespecified_checks", {})
        rows.append(ClaimRow(
            "temporal-control-observable",
            "Public temporal-history prediction gain is a discriminant observational measure of Control.",
            _bool_status(bool(temporal.get("temporal_control_confirmed")), checks),
            _fmt_bool_checks(checks),
            "validation/temporal-control.json",
            "synthetic observational measurement",
        ))

    contextual = get("validation/contextual-control-observable.json")
    if contextual:
        checks = contextual.get("checks", {})
        rows.append(ClaimRow(
            "contextual-control-observable",
            "Matched public-history likelihood contrast is a family-invariant observational measure of Control.",
            _bool_status(bool(contextual.get("contextual_control_observable_confirmed")), checks),
            f"cross_family_control_gap={contextual.get('cross_family_control_gap'):.3f}; " + _fmt_bool_checks(checks),
            "validation/contextual-control-observable.json",
            "two synthetic policy families",
        ))

    eff = get("validation/effective-chaos-validation.json")
    if eff:
        checks = eff.get("prespecified_checks", {})
        rows.append(ClaimRow(
            "effective-chaos",
            "Value-floor-weighted effective surprisal is a cross-family discriminant observational measure of Chaos.",
            _bool_status(bool(eff.get("effective_chaos_construct_confirmed")), checks),
            _fmt_bool_checks(checks),
            "validation/effective-chaos-validation.json",
            "two synthetic policy families",
        ))

    cc = get("validation/chaos-control-decomposition.json")
    if cc:
        checks = cc.get("prespecified_checks", {})
        rows.append(ClaimRow(
            "chaos-control-decomposition",
            "Conditioning on public history universally separates Control-linked surprise from Chaos-linked residual surprise.",
            _bool_status(bool(cc.get("chaos_control_entanglement_decomposition_supported")), checks),
            _fmt_bool_checks(checks),
            "validation/chaos-control-decomposition.json",
            "two synthetic policy families",
        ))

    ps = get("validation/pressure-surprise-decomposition.json")
    if ps:
        checks = ps.get("prespecified_checks", {})
        rows.append(ClaimRow(
            "pressure-surprise-suppression",
            "Public Pressure exposure universally explains the negative Pressure/effective-surprisal association and improves Chaos discrimination.",
            _bool_status(bool(ps.get("pressure_suppression_mechanism_supported")), checks),
            _fmt_bool_checks(checks),
            "validation/pressure-surprise-decomposition.json",
            "two synthetic policy families",
        ))

    panel = get("validation/family-invariant-panel.json")
    if panel:
        selected = panel.get("selected_invariant_components", [])
        coverage = panel.get("axis_coverage", {})
        rows.append(ClaimRow(
            "family-invariant-pressure-panel",
            "At least one conservative label-free Pressure observable is stable across both policy families.",
            "confirmed" if coverage.get("pressure") else "failed",
            "selected: " + ", ".join(coverage.get("pressure", [])),
            "validation/family-invariant-panel.json",
            "two synthetic policy families",
        ))
        for axis in ("control", "chaos"):
            rows.append(ClaimRow(
                f"family-invariant-{axis}-panel",
                f"A conservative label-free {axis.title()} observable is currently supported across both policy families.",
                "confirmed" if coverage.get(axis) else "unresolved",
                "selected: " + (", ".join(coverage.get(axis, [])) or "none"),
                "validation/family-invariant-panel.json",
                "two synthetic policy families",
            ))

    transfer = get("validation/family-transfer-grid-summary.json")
    if transfer:
        conclusion = str(transfer.get("conclusion", ""))
        rows.append(ClaimRow(
            "implementation-invariant-supervised-coordinates",
            "Supervised PCC coordinate recovery transfers invariantly between the Score and Independent policy families.",
            "failed" if "not implementation-invariant" in conclusion else "partial",
            conclusion or "no explicit conclusion",
            "validation/family-transfer-grid-summary.json",
            "cross-family supervised transfer",
        ))

    mixed = get("validation/mixed-recovery.json")
    if mixed:
        checks = mixed.get("prespecified_checks", {})
        status = "confirmed" if checks and all(checks.values()) else ("partial" if any(checks.values()) else "failed")
        rows.append(ClaimRow(
            "synthetic-mixture-identifiability",
            "Continuous synthetic PCC mixture weights are identifiable above the prespecified baselines within the development family.",
            status,
            f"relative_mae_improvement_over_action_frequency={mixed.get('relative_mae_improvement_over_action_frequency', 0):.3f}; " + _fmt_bool_checks(checks),
            "validation/mixed-recovery.json",
            "synthetic mixture recovery; not human construct validity",
        ))

    repro = get("validation/reproducibility-manifest.json")
    if repro:
        rows.append(ClaimRow(
            "reproducibility-audit",
            "The frozen synthetic validation bundle is complete and reproducibility-audit ready.",
            "confirmed" if repro.get("reproducibility_ready") else "failed",
            f"reproducibility_ready={repro.get('reproducibility_ready')}; present={repro.get('frozen_validation',{}).get('present_count')}/{repro.get('frozen_validation',{}).get('requested_count')}",
            "validation/reproducibility-manifest.json",
            "engineering/reproducibility",
        ))

    counts = {s: sum(r.status == s for r in rows) for s in ("confirmed", "partial", "failed", "unresolved")}
    return {
        "schema_version": 1,
        "purpose": "Publication-oriented status summary of frozen synthetic PCC Poker claims. Human-data claims are intentionally excluded.",
        "status_definitions": {
            "confirmed": "All direct preregistered acceptance criteria represented by the source artifact passed.",
            "partial": "The primary confirmation failed, but one or more prespecified component checks passed.",
            "failed": "The represented claim failed its direct acceptance criterion without qualifying component support sufficient for partial status.",
            "unresolved": "No conservative cross-family observable is currently selected for this claim/axis.",
        },
        "counts": counts,
        "claims": [asdict(r) for r in rows],
        "missing_sources": missing,
    }


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# PCC Poker research status",
        "",
        "> Frozen synthetic evidence only. This table does not make claims about human poker behavior.",
        "",
        f"**Summary:** {report['counts']['confirmed']} confirmed · {report['counts']['partial']} partial · {report['counts']['failed']} failed · {report['counts']['unresolved']} unresolved",
        "",
        "| Claim | Status | Evidence | Scope | Source |",
        "|---|---|---|---|---|",
    ]
    for row in report["claims"]:
        vals = [row["claim"], row["status"].upper(), row["evidence"], row["scope"], f"`{row['source']}`"]
        vals = [str(v).replace("|", "\\|").replace("\n", " ") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")
    if report["missing_sources"]:
        lines += ["", "## Missing expected sources", ""] + [f"- `{p}`" for p in report["missing_sources"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_research_status(
    *,
    root: str | Path = ".",
    json_output: str | Path = "validation/research-status.json",
    csv_output: str | Path = "validation/research-status.csv",
    markdown_output: str | Path = "validation/RESEARCH_STATUS.md",
) -> dict:
    root = Path(root).resolve()
    report = build_research_status(root)
    jpath = root / json_output if not Path(json_output).is_absolute() else Path(json_output)
    cpath = root / csv_output if not Path(csv_output).is_absolute() else Path(csv_output)
    mpath = root / markdown_output if not Path(markdown_output).is_absolute() else Path(markdown_output)
    for p in (jpath, cpath, mpath):
        p.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with cpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["claim_id", "claim", "status", "evidence", "scope", "source"])
        writer.writeheader()
        writer.writerows(report["claims"])
    _write_markdown(report, mpath)
    return report
