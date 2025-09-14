#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuzzy_sugeno_2in2out.py
---------------------------------
Two-input / Two-output Sugeno (zero-order) fuzzy controller
for a massage-chair-like demo.

Why is the code relatively short?
- Sugeno zero-order rules return constants -> no defuzz of output MFs.
- Product inference + weighted average is compact to implement.
- We encapsulate logic in small classes (Term, SugenoRule, SugenoSystem).

Default behavior:
- If run as a script without arguments, it will look for
  'Muscle_Tension_DataSample.csv' in the current directory,
  read columns EMG_RMS_mV and Pressure_N, and produce
  'Muscle_Tension_DataSample_withOutputs.csv'.

Usage examples:
  # 1) Auto-use Muscle_Tension_DataSample.csv in the same folder
  python fuzzy_sugeno_2in2out.py

  # 2) Specify custom input/output and column names
  python fuzzy_sugeno_2in2out.py --in your_input.csv --out results.csv \
      --emg-col EMG_RMS_mV --press-col Pressure_N

Library usage:
  from fuzzy_sugeno_2in2out import build_massage_chair_sugeno, evaluate_dataframe
  fis = build_massage_chair_sugeno()
  out_df = evaluate_dataframe(fis, df, emg_col="EMG_RMS_mV", press_col="Pressure_N")

python fuzzy_sugeno_2in2out.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# Membership function definitions
# -------------------------------

def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function. Returns degree in [0,1]."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    # Rising edge
    idx = (a < x) & (x <= b)
    if b != a:
        y[idx] = (x[idx] - a) / (b - a)
    # Falling edge
    idx = (b < x) & (x < c)
    if c != b:
        y[idx] = (c - x[idx]) / (c - b)
    # Peak point
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership function. Returns degree in [0,1]."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    # Rising
    idx = (a < x) & (x <= b)
    if b != a:
        y[idx] = (x[idx] - a) / (b - a)
    # Plateau
    idx2 = (b < x) & (x <= c)
    y[idx2] = 1.0
    # Falling
    idx3 = (c < x) & (x < d)
    if d != c:
        y[idx3] = (d - x[idx3]) / (d - c)
    # Exact edges
    y[(x == b) | (x == c)] = 1.0
    return np.clip(y, 0.0, 1.0)


# -------------------------------
# Linguistic terms and Sugeno FIS
# -------------------------------

@dataclass
class Term:
    """Linguistic term for an input variable, with a membership function."""
    name: str
    func: Callable[[np.ndarray], np.ndarray]

    def mu(self, x_scalar: float) -> float:
        return float(self.func(np.array([x_scalar], dtype=float))[0])


@dataclass
class SugenoRule:
    """Sugeno (zero-order) rule: IF antecedents THEN outputs=constants.
    - antecedents: list of (var_name, term_name)
    - consequents: dict {output_name: constant_value}
    """
    antecedents: List[Tuple[str, str]]
    consequents: Dict[str, float]


class SugenoSystem:
    """Sugeno zero-order FIS (product inference, weighted average)."""

    def __init__(self) -> None:
        self.input_terms: Dict[str, Dict[str, Term]] = {}
        self.outputs: List[str] = []
        self.rules: List[SugenoRule] = []

    # ---- Build ----
    def add_input_variable(self, var_name: str, terms: Dict[str, Term]) -> None:
        self.input_terms[var_name] = terms

    def set_outputs(self, output_names: List[str]) -> None:
        self.outputs = list(output_names)

    def add_rule(self, rule: SugenoRule) -> None:
        for out in self.outputs:
            if out not in rule.consequents:
                raise ValueError(f"Rule missing consequent for '{out}'")
        self.rules.append(rule)

    # ---- Inference ----
    def _firing_strength(self, input_values: Dict[str, float], rule: SugenoRule) -> float:
        w = 1.0
        for var_name, term_name in rule.antecedents:
            term = self.input_terms[var_name][term_name]
            w *= term.mu(input_values[var_name])
            if w == 0.0:
                break
        return w

    def evaluate(self, input_values: Dict[str, float]) -> Dict[str, float]:
        num = {out: 0.0 for out in self.outputs}
        den = 0.0
        for r in self.rules:
            w = self._firing_strength(input_values, r)
            if w <= 0.0:
                continue
            den += w
            for out in self.outputs:
                num[out] += w * r.consequents[out]
        if den == 0.0:
            return {out: 0.0 for out in self.outputs}
        return {out: num[out] / den for out in self.outputs}


# -------------------------------
# Build a concrete controller
# -------------------------------

def build_massage_chair_sugeno() -> SugenoSystem:
    """Create a Sugeno controller with 2 inputs (EMG_RMS_mV, Pressure_N) and
    2 outputs (Intensity_pct, Frequency_Hz). Nine rules (3x3)."""
    fis = SugenoSystem()

    # Input 1: EMG RMS (mV) -- approximate ranges (tune for your data)
    fis.add_input_variable("EMG_RMS_mV", {
        "Low":  Term("Low",  lambda x: trapmf(x, -1.0, 0.0, 0.8, 1.4)),
        "Med":  Term("Med",  lambda x: trimf(x, 0.8, 1.7, 2.6)),
        "High": Term("High", lambda x: trapmf(x, 2.2, 2.8, 5.0, 8.0)),
    })

    # Input 2: Pressure (N) -- proxy for posture/tension
    fis.add_input_variable("Pressure_N", {
        "Low":  Term("Low",  lambda x: trapmf(x, 10, 20, 26, 32)),
        "Med":  Term("Med",  lambda x: trimf(x, 28, 36, 46)),
        "High": Term("High", lambda x: trapmf(x, 42, 50, 70, 85)),
    })

    fis.set_outputs(["Intensity_pct", "Frequency_Hz"])

    # Helper for concise rule definitions
    def R(emg_term: str, p_term: str, intensity: float, freq: float) -> None:
        fis.add_rule(SugenoRule(
            antecedents=[("EMG_RMS_mV", emg_term), ("Pressure_N", p_term)],
            consequents={"Intensity_pct": float(intensity), "Frequency_Hz": float(freq)}
        ))

    # Expert heuristic: when tension (EMG/Pressure) is high -> gentler (lower intensity/frequency)
    R("High", "High",  35,  8)
    R("High", "Med",   45, 10)
    R("High", "Low",   55, 12)

    R("Med",  "High",  45, 10)
    R("Med",  "Med",   60, 13)
    R("Med",  "Low",   70, 14)

    R("Low",  "High",  55, 12)
    R("Low",  "Med",   70, 14)
    R("Low",  "Low",   85, 16)

    return fis


# -------------------------------
# Batch evaluation utility
# -------------------------------

def evaluate_dataframe(fis: SugenoSystem, df: pd.DataFrame,
                       emg_col: str = "EMG_RMS_mV", press_col: str = "Pressure_N") -> pd.DataFrame:
    """Apply the Sugeno controller to a DataFrame and return a copy with outputs added.
    Required columns: emg_col, press_col.
    """
    missing = [c for c in (emg_col, press_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Available columns: {list(df.columns)}")

    out_I = []
    out_F = []
    for emg, p in zip(df[emg_col].values, df[press_col].values):
        res = fis.evaluate({"EMG_RMS_mV": float(emg), "Pressure_N": float(p)})
        out_I.append(res["Intensity_pct"])
        out_F.append(res["Frequency_Hz"])
    out_df = df.copy()
    out_df["Intensity_pct"] = np.round(out_I, 3)
    out_df["Frequency_Hz"]  = np.round(out_F, 3)
    return out_df


# -------------------------------
# CLI
# -------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2-input/2-output Sugeno fuzzy controller (massage chair demo)")
    p.add_argument("--in", dest="in_path", type=str, default=None,
                   help="Input CSV path. By default, tries 'Muscle_Tension_DataSample.csv' in CWD.")
    p.add_argument("--out", dest="out_path", type=str, default=None,
                   help="Output CSV path. Default is '<input>_withOutputs.csv'.")
    p.add_argument("--emg-col", type=str, default="EMG_RMS_mV",
                   help="Column for EMG RMS (default: EMG_RMS_mV).")
    p.add_argument("--press-col", type=str, default="Pressure_N",
                   help="Column for Pressure (default: Pressure_N).")
    return p


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    # Default input/output behavior
    in_path = args.in_path or "Muscle_Tension_DataSample.csv"
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Input CSV '{in_path}' not found. "
            "Place 'Muscle_Tension_DataSample.csv' in this folder or pass --in your_file.csv"
        )

    # Read data
    df = pd.read_csv(in_path)

    # Heuristic auto-detect if the user uses different case/columns
    # Try common aliases
    emg_col = args.emg_col if args.emg_col in df.columns else None
    press_col = args.press_col if args.press_col in df.columns else None

    if emg_col is None:
        for cand in ["emg_rms_mv", "emg_rms", "emg", "EMG", "EMG_RMS_mV"]:
            if cand in df.columns:
                emg_col = cand
                break
    if press_col is None:
        for cand in ["pressure_n", "pressure", "PRESSURE_N", "Pressure_N"]:
            if cand in df.columns:
                press_col = cand
                break

    if emg_col is None or press_col is None:
        raise ValueError("Could not find required columns for EMG and Pressure. "
                         f"Available columns: {list(df.columns)}. "
                         "Specify with --emg-col and --press-col.")

    # Build controller and evaluate
    fis = build_massage_chair_sugeno()
    out_df = evaluate_dataframe(fis, df, emg_col=emg_col, press_col=press_col)

    # Output path
    out_path = args.out_path or os.path.splitext(in_path)[0] + "_withOutputs.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Read: {in_path}")
    print(f"[OK] Columns used -> EMG: '{emg_col}', Pressure: '{press_col}'")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
