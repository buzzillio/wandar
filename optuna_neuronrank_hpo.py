#!/usr/bin/env python3
"""Hyperparameter optimization runner for Wanda pruning.

This script wraps ``main.py`` with an Optuna study to tune the Wanda
scoring multipliers (--w-alpha and --w-beta). It keeps the base command 
fixed and searches over the specified ranges using discrete steps of 0.2. 
Lower perplexity is better, so the Optuna objective minimizes the Wikitext 
perplexity reported by ``main.py``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import optuna

BASE_COMMAND = [
    sys.executable,
    "main.py",
    "--model",
    "baffo32/decapoda-research-llama-7B-hf",
    "--prune_method",
    "wanda",
    "--sparsity_ratio",
    "0.5",
    "--sparsity_type",
    "unstructured",
    "--nsamples",
    "128",
]

PPL_REGEX = re.compile(r"wikitext perplexity\s+([0-9]+\.?[0-9]*)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna HPO for Wanda")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="wanda_hpo",
        help="Name of the Optuna study.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///wanda.db). If omitted, an in-memory study is used.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="out/wanda_hpo",
        help="Base directory where trial outputs will be written.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Global timeout for the study in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed passed to the Optuna sampler.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove any existing save directory before starting the study.",
    )
    return parser.parse_args()


def build_command(save_path: Path, w_alpha: float, w_beta: float) -> list[str]:
    command = list(BASE_COMMAND)
    command.extend(
        [
            "--w-alpha",
            f"{w_alpha}",
            "--w-beta",
            f"{w_beta}",
            "--save",
            str(save_path),
        ]
    )
    return command


def run_trial(command: list[str], env: Optional[dict[str, str]] = None) -> float:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = process.stdout
    stderr = process.stderr

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    match = PPL_REGEX.search(stdout)
    if not match:
        raise RuntimeError(
            "Unable to parse Wikitext perplexity from command output.\n" f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    return float(match.group(1))


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    if args.clean and save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed) if args.seed is not None else optuna.samplers.TPESampler()

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    def objective(trial: optuna.trial.Trial) -> float:
        # Search ranges: 0.2 to 1.0 with step 0.2
        w_alpha = trial.suggest_float("w_alpha", 0.2, 1.0, step=0.2)
        w_beta = trial.suggest_float("w_beta", 0.2, 1.0, step=0.2)

        trial_save_dir = save_dir / f"trial_{trial.number:04d}"
        trial_save_dir.mkdir(parents=True, exist_ok=True)

        command = build_command(
            trial_save_dir,
            w_alpha=w_alpha,
            w_beta=w_beta,
        )

        print(
            f"[Trial {trial.number}] Running: w_alpha={w_alpha}, w_beta={w_beta}"
        )
        ppl = run_trial(command, env=env)
        print(f"[Trial {trial.number}] Wikitext perplexity: {ppl:.4f}")
        return ppl

    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    except KeyboardInterrupt:
        print("Interrupted by user. Returning current best result.")

    if study.best_trial is None:
        print("No successful trials completed.")
        return

    best = study.best_trial
    print("\n=== Best Trial ===")
    print(f"Trial #: {best.number}")
    print(f"Perplexity: {best.value:.4f}")
    for key, value in best.params.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
