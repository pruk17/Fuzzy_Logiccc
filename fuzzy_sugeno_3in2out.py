# fuzzy_sugeno_3in2out_with_feeling_fixed.py
# -------------------------------------------
# Sugeno (zero-order) fuzzy controller with 3 inputs and 2 outputs, plus verbose printing.
# Inputs:
#   - EMG_RMS_mV   (numeric) -> Low/Med/High
#   - Pressure_N   (numeric) -> Low/Med/High
#   - Pain_Score   (numeric, derived from 'feeling' text) -> Low/Med/High
# Outputs:
#   - Intensity_pct (0..100)
#   - Frequency_Hz  (~8..16)
#

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ------------- Membership functions -------------

def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    idx = (a < x) & (x <= b)
    if b != a:
        y[idx] = (x[idx] - a) / (b - a)
    idx = (b < x) & (x < c)
    if c != b:
        y[idx] = (c - x[idx]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)

def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    idx = (a < x) & (x <= b)
    if b != a:
        y[idx] = (x[idx] - a) / (b - a)
    idx2 = (b < x) & (x <= c)
    y[idx2] = 1.0
    idx3 = (c < x) & (x < d)
    if d != c:
        y[idx3] = (d - x[idx3]) / (d - c)
    y[(x == b) | (x == c)] = 1.0
    return np.clip(y, 0.0, 1.0)

# ------------- Linguistic terms & FIS -------------

@dataclass
class Term:
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    def mu(self, x_scalar: float) -> float:
        return float(self.func(np.array([x_scalar], dtype=float))[0])

@dataclass
class SugenoRule:
    antecedents: List[Tuple[str, str]]
    consequents: Dict[str, float]

class SugenoSystem:
    def __init__(self) -> None:
        self.input_terms: Dict[str, Dict[str, Term]] = {}
        self.outputs: List[str] = []
        self.rules: List[SugenoRule] = []

    def add_input_variable(self, var_name: str, terms: Dict[str, Term]) -> None:
        self.input_terms[var_name] = terms

    def set_outputs(self, output_names: List[str]) -> None:
        self.outputs = list(output_names)

    def add_rule(self, rule: SugenoRule) -> None:
        for out in self.outputs:
            if out not in rule.consequents:
                raise ValueError(f"Rule missing consequent for '{out}'")
        self.rules.append(rule)

    def _firing_strength(self, input_values: Dict[str, float], rule: SugenoRule) -> float:
        w = 1.0
        for var_name, term_name in rule.antecedents:
            term = self.input_terms[var_name][term_name]
            w *= term.mu(input_values[var_name])
            if w == 0.0:
                break
        return w

    def evaluate(self, input_values: Dict[str, float]):
        num = {out: 0.0 for out in self.outputs}
        den = 0.0
        details: List[Tuple[float, SugenoRule]] = []
        for r in self.rules:
            w = self._firing_strength(input_values, r)
            if w > 0.0:
                details.append((w, r))
                den += w
                for out in self.outputs:
                    num[out] += w * r.consequents[out]
        details.sort(key=lambda x: -x[0])
        if den == 0.0:
            return {out: 0.0 for out in self.outputs}, details
        return {out: num[out] / den for out in self.outputs}, details

# ------------- Build controller -------------

def build_massage_chair_sugeno_3in() -> SugenoSystem:
    fis = SugenoSystem()

    fis.add_input_variable("EMG_RMS_mV", {
        "Low":  Term("Low",  lambda x: trapmf(x, -1.0, 0.0, 0.8, 1.4)),
        "Med":  Term("Med",  lambda x: trimf(x, 0.8, 1.7, 2.6)),
        "High": Term("High", lambda x: trapmf(x, 2.2, 2.8, 5.0, 8.0)),
    })
    fis.add_input_variable("Pressure_N", {
        "Low":  Term("Low",  lambda x: trapmf(x, 10, 20, 26, 32)),
        "Med":  Term("Med",  lambda x: trimf(x, 28, 36, 46)),
        "High": Term("High", lambda x: trapmf(x, 42, 50, 70, 85)),
    })
    fis.add_input_variable("Pain_Score", {
        "Low":  Term("Low",  lambda x: trapmf(x, -10,  0,  20, 35)),
        "Med":  Term("Med",  lambda x: trimf(x,   25, 50, 75)),
        "High": Term("High", lambda x: trapmf(x,  60, 75, 100, 120)),
    })

    fis.set_outputs(["Intensity_pct", "Frequency_Hz"])

    base = {
        ("High","High"):(35,8), ("High","Med"):(45,10), ("High","Low"):(55,12),
        ("Med","High") :(45,10),("Med","Med"):(60,13), ("Med","Low"):(70,14),
        ("Low","High") :(55,12),("Low","Med"):(70,14), ("Low","Low"):(85,16),
    }
    pain_mod = {"Low":(+5.0,+1.0), "Med":(0.0,0.0), "High":(-15.0,-2.0)}

    for (e,p),(I0,F0) in base.items():
        for pain, (dI,dF) in pain_mod.items():
            I = float(np.clip(I0+dI, 0, 100))
            F = float(np.clip(F0+dF, 0, 100))
            fis.add_rule(SugenoRule(
                antecedents=[("EMG_RMS_mV",e),("Pressure_N",p),("Pain_Score",pain)],
                consequents={"Intensity_pct": I, "Frequency_Hz": F}
            ))
    return fis

# ------------- Feeling mapping (English only) -------------

FEELING_MAP = {
    "very relaxed": 10,
    "relaxed": 20,
    "calm": 20,
    "ok": 50,
    "normal": 50,
    "neutral": 50,
    "fine": 50,
    "little pain": 40,
    "mild pain": 40,
    "pain": 70,
    "severe pain": 90,
    "very pain": 90,  # legacy
}

def feeling_to_score(text: Optional[str], default: float = 50.0) -> float:
    if text is None:
        return float(default)
    t = str(text).strip().lower()
    return float(FEELING_MAP.get(t, default))

# ------------- Verbose evaluator -------------

class TopFmt(str, Enum):
    long = "long"       # e.g., (EMG=Med, P=Med, Pain=Low)
    compact = "compact" # e.g., (E=M, P=M, Pa=L)

def _fmt_ant(a: str, b: str, c: str, fmt: TopFmt = TopFmt.compact) -> str:
    if fmt == TopFmt.long:
        return f"(EMG={a}, P={b}, Pain={c})"
    abbr = {"Low": "L", "Med": "M", "High": "H"}
    return f"(E={abbr.get(a,a[:1])}, P={abbr.get(b,b[:1])}, Pa={abbr.get(c,c[:1])})"

def _wrap_parts(parts: List[str], max_width: int, indent: int = 4, prefix: str = "Top: ") -> List[str]:
    lines: List[str] = []
    cur = " " * indent + prefix
    sep = " | "
    for p in parts:
        add = ("" if cur.strip().endswith(prefix.strip()) else sep) + p
        if len(cur) + len(add) <= max_width:
            cur += add
        else:
            lines.append(cur)
            cur = " " * indent + prefix + p
    lines.append(cur)
    return lines

def _top_k_rules(details: List[Tuple[float, SugenoRule]], k: int = 3) -> List[Tuple[float, Tuple[str, str, str]]]:
    top: List[Tuple[float, Tuple[str, str, str]]] = []
    for w, r in details[:k]:
        ants = [a[1] for a in r.antecedents]
        while len(ants) < 3:
            ants.append("-")
        top.append((w, (ants[0], ants[1], ants[2])))
    return top

def evaluate_csv_with_print(
    fis: SugenoSystem,
    df: pd.DataFrame,
    emg_col: str = "EMG_RMS_mV",
    press_col: str = "Pressure_N",
    feel_col: str = "feeling",
    out_path: Optional[str] = None,
    *,
    dataset_name: Optional[str] = None,
    topk: int = 3,
    min_w: float = 0.10,
    max_topline_chars: Optional[int] = None,
    block: int = 25,
    topfmt: TopFmt = TopFmt.compact,
) -> pd.DataFrame:
    """Evaluate CSV row-by-row and print a neat, readable table to terminal."""
    pain_scores: List[float] = []
    out_I: List[float] = []
    out_F: List[float] = []

    # column widths
    Wt, Wemg, Wp, Wfeel, Wpain, Wint, Wfreq = 7, 9, 12, 16, 10, 11, 10

    # de-duplicate rows (avoid noisy repeated prints if dataset has duplicates)
    subset_cols = [c for c in ["time_s", emg_col, press_col, feel_col] if c in df.columns]
    if subset_cols:
        df = (
            df.sort_values(subset_cols)
              .drop_duplicates(subset=subset_cols, keep="first")
              .reset_index(drop=True)
        )

    # auto-detect terminal width
    if max_topline_chars is None:
        try:
            cols = shutil.get_terminal_size((120, 20)).columns
        except Exception:
            cols = 120
        max_topline_chars = max(60, cols - 2)

    # ---------- CONFIG HEADER (experiment parameters) ----------
    cfg_lines = [
        "Run configuration:",
        f"  Dataset            : {dataset_name or '<DataFrame>'}",
        f"  Columns            : EMG='{emg_col}', Pressure='{press_col}', Feeling='{feel_col}'",
        f"  Rules/Outputs      : {len(fis.rules)} rules -> {', '.join(fis.outputs)}",
        f"  Printing params    : TOPK={topk}, MIN_W={min_w:.3f}, MAX_TOPLINE={max_topline_chars}, BLOCK={block}, TOPFMT={topfmt}",
        f"  Feeling map keys   : {', '.join(sorted(FEELING_MAP.keys()))}",
    ]
    max_len = max(len(s) for s in cfg_lines)
    bar_cfg = "-" * max_len
    print(bar_cfg)
    for s in cfg_lines:
        print(s)
    print(bar_cfg)

    def header() -> None:
        line = (
            f"{'t[s]':>{Wt}} "
            f"{'EMG[mV]':>{Wemg}} "
            f"{'Pressure[N]':>{Wp}} "
            f"{'Feeling':<{Wfeel}} "
            f"{'Pain':>{Wpain}} "
            f"{'Intensity%':>{Wint}} "
            f"{'Freq[Hz]':>{Wfreq}}"
        )
        bar = "-" * len(line)
        print(bar)
        print(line)
        print(bar)

    def trunc(s: str, width: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= width else (s[:max(0, width-1)] + "…")

    # initial header
    header()

    for idx, row in df.iterrows():
        emg = float(row[emg_col])
        press = float(row[press_col])
        feeling = row[feel_col] if (feel_col in df.columns) else ""
        pain = feeling_to_score(feeling)

        outputs, details = fis.evaluate({"EMG_RMS_mV": emg, "Pressure_N": press, "Pain_Score": pain})
        I = float(outputs["Intensity_pct"])
        F = float(outputs["Frequency_Hz"])

        pain_scores.append(pain)
        out_I.append(I)
        out_F.append(F)

        if idx and (idx % block == 0):
            header()

        t_str = f"{row['time_s']:.1f}" if 'time_s' in df.columns else f"{idx}"
        line = (
            f"{t_str:>{Wt}} "
            f"{emg:>{Wemg}.3f} "
            f"{press:>{Wp}.2f} "
            f"{trunc(str(feeling), Wfeel):<{Wfeel}} "
            f"{pain:>{Wpain}.1f} "
            f"{I:>{Wint}.2f} "
            f"{F:>{Wfreq}.2f}"
        )
        print(line)

        # Top-k rules (filtered + wrapped)
        tops = _top_k_rules(details, k=topk)
        parts = [f"w={w:.3f} " + _fmt_ant(a, b, c, topfmt) for w, (a, b, c) in tops if w >= min_w]
        if parts:
            for wrapped_line in _wrap_parts(parts, max_width=max_topline_chars, indent=4, prefix="Top: "):
                print(wrapped_line)

    out_df = df.copy()
    out_df["Pain_Score"] = np.round(pain_scores, 2)
    out_df["Intensity_pct"] = np.round(out_I, 3)
    out_df["Frequency_Hz"] = np.round(out_F, 3)

    if out_path:
        out_df.to_csv(out_path, index=False)
        print("-" * 72)
        print(f"[OK] Wrote outputs to: {out_path}")
    return out_df

# ------------- CLI -------------

def _make_parser():
    p = argparse.ArgumentParser(description="3-input/2-output Sugeno fuzzy controller with feeling text input (verbose)")
    p.add_argument("--in", dest="in_path", type=str, default="Muscle_Tension_Data.csv",
                   help="Input CSV path (default: Muscle_Tension_Data.csv).")
    p.add_argument("--out", dest="out_path", type=str, default=None,
                   help="Output CSV path (default: Muscle_Tension_Output.csv).")
    p.add_argument("--emg-col", type=str, default="EMG_RMS_mV",
                   help="Column for EMG (default: EMG_RMS_mV).")
    p.add_argument("--press-col", type=str, default="Pressure_N",
                   help="Column for Pressure (default: Pressure_N).")
    p.add_argument("--feel-col", type=str, default="feeling",
                   help="Column for feeling text (default: feeling).")
    # printing / filtering params
    p.add_argument("--topk", type=int, default=3, help="How many top rules to show (default: 3).")
    p.add_argument("--min-w", type=float, default=0.10, help="Min firing strength to display (default: 0.10).")
    p.add_argument("--max-topline", type=int, default=None,
                   help="Max chars for each 'Top:' line; if omitted, auto-detect terminal width.")
    p.add_argument("--block", type=int, default=25, help="Header repeat interval (rows) (default: 25).")
    p.add_argument("--topfmt", type=str, choices=[v.value for v in TopFmt], default=TopFmt.compact.value,
                   help="Format of Top antecedents: 'compact' or 'long' (default: compact).")
    return p

def main():
    parser = _make_parser()
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"Input CSV '{args.in_path}' not found.")

    df = pd.read_csv(args.in_path)
    for col in [args.emg_col, args.press_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Columns: {list(df.columns)}")

    fis = build_massage_chair_sugeno_3in()
    out_path = args.out_path or "Muscle_Tension_Output.csv"

    evaluate_csv_with_print(
        fis, df,
        emg_col=args.emg_col, press_col=args.press_col, feel_col=args.feel_col,
        out_path=out_path,
        dataset_name=os.path.basename(args.in_path),
        topk=args.topk,
        min_w=args.min_w,
        max_topline_chars=args.max_topline,
        block=args.block,
        topfmt=TopFmt(args.topfmt),
    )

if __name__ == "__main__":
    main()
