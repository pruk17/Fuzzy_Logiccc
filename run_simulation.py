# run_simulation.py
# Closed-loop simulation to satisfy assignment item (2):
# - Uses your Sugeno controller (2 in / 2 out)
# - Simulates a simple "muscle tension" plant over time
# - Compares Fuzzy vs Baseline (constant intensity/frequency)
# - Writes results to CSV; optional matplotlib plots

import numpy as np
import pandas as pd
import os

# 1) import your controller module
from fuzzy_sugeno_2in2out import build_massage_chair_sugeno

# 2) simple plant model (toy but illustrative)
# State: T_k in [0..100] = muscle tension level (higher = more tense)
# Inputs: Intensity_pct (0..100), Frequency_Hz (typ. 8..16)
# Update:
#   T_{k+1} = T_k
#            - a * I_eff                      (massage relax effect)
#            + b * max(0, I_eff - I_comf)     (over-intensity soreness penalty)
#            - c * F_eff                      (gentle frequency helps a bit)
#            + d * disturbance_k              (random user movement)
#            + noise
# where I_eff, F_eff are normalized
def step_plant(T, intensity_pct, freq_hz, a=0.06, b=0.03, c=0.02,
               I_comf=60.0, amb=0.0, rng=None):
    rng = rng or np.random.default_rng()
    I_eff = np.clip(intensity_pct, 0, 100) / 100.0       # 0..1
    F_eff = np.clip((freq_hz - 8) / (16 - 8), 0.0, 1.0)  # map 8..16 Hz -> 0..1
    disturbance = rng.normal(0.0, 0.6) + amb
    noise = rng.normal(0.0, 0.3)

    d_relax = -a * I_eff
    d_over  =  b * max(0.0, intensity_pct - I_comf) / 100.0
    d_freq  = -c * F_eff

    T_next = T + d_relax + d_over + d_freq + 0.02*disturbance + noise*0.01
    return float(np.clip(T_next, 0.0, 100.0))

def simulate(controller, T0=60.0, seconds=300, baseline=False,
             baseline_I=60.0, baseline_F=12.0, seed=7):
    rng = np.random.default_rng(seed)
    T = T0
    rows = []
    for t in range(seconds):
        # build inputs (EMG, Pressure) synthetically from T
        # EMG_RMS ~ 0.02*T + noise; Pressure_N ~ 30 + 0.1*T + noise
        emg = max(0.0, 0.02*T + rng.normal(0, 0.25))
        pressure = max(10.0, 30.0 + 0.1*T + rng.normal(0, 0.9))

        if baseline:
            I, F = baseline_I, baseline_F
        else:
            out = controller.evaluate({"EMG_RMS_mV": emg, "Pressure_N": pressure})
            I, F = out["Intensity_pct"], out["Frequency_Hz"]

        T_next = step_plant(T, I, F, rng=rng)
        rows.append({
            "t_s": t,
            "EMG_RMS_mV": round(emg, 3),
            "Pressure_N": round(pressure, 2),
            "Intensity_pct": round(I, 3),
            "Frequency_Hz": round(F, 3),
            "Tension": round(T, 3),
            "Tension_next": round(T_next, 3),
            "mode": "baseline" if baseline else "fuzzy"
        })
        T = T_next
    return pd.DataFrame(rows)

def main():
    fis = build_massage_chair_sugeno()

    # Sim A: Fuzzy closed-loop
    df_fuzzy = simulate(fis, T0=60.0, seconds=300, baseline=False, seed=11)

    # Sim B: Baseline (no fuzzy) with fixed moderate settings
    df_base  = simulate(fis, T0=60.0, seconds=300, baseline=True,
                        baseline_I=60.0, baseline_F=12.0, seed=11)

    # Merge for comparison
    df_all = pd.concat([df_fuzzy, df_base], ignore_index=True)
    out_csv = "massage_chair_simulation_results.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    # Quick KPIs for report
    def kpi(df):
        return {
            "Tension_peak": df["Tension"].max(),
            "Tension_mean": df["Tension"].mean(),
            "Below30_ratio": (df["Tension"] <= 30).mean(),  # fraction of time in comfort zone
            "Intensity_mean": df["Intensity_pct"].mean(),
        }
    kpi_fuzzy = kpi(df_fuzzy)
    kpi_base  = kpi(df_base)

    print("\n=== KPIs ===")
    print("Fuzzy :", kpi_fuzzy)
    print("Base  :", kpi_base)

    # Optional: quick plots (uncomment if you want graphs)
    # import matplotlib.pyplot as plt
    # plt.figure(); plt.plot(df_fuzzy["t_s"], df_fuzzy["Tension"], label="Fuzzy")
    # plt.plot(df_base["t_s"], df_base["Tension"], label="Baseline")
    # plt.xlabel("time (s)"); plt.ylabel("Tension"); plt.legend(); plt.title("Tension vs time")
    # plt.tight_layout(); plt.savefig("plot_tension.png")

if __name__ == "__main__":
    main()
