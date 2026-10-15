"""
Computational analysis for the chapter:
"Powering Intelligence, Pricing the Planet: Environmental Impacts of Digital
Infrastructure and Data Centers and the Leadership Agenda for Sustainable AI"

Six linked models (all reproducible, seed fixed):
  M1  Scenario model of global data-centre electricity, CO2 and water, 2024-2035
  M2  Monte Carlo uncertainty propagation (N = 20,000)
  M3  Variance-based (Sobol) global sensitivity analysis (Saltelli/Jansen estimators)
  M4  Carbon-aware workload scheduling (linear programme, HiGHS)
  M5  Hardware-refresh lifecycle model (embodied vs operational carbon, e-waste)
  M6  Abatement waterfall + TOPSIS multi-criteria ranking of leadership levers
Outputs: figs/*.png, tables/*.csv, results.json
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch
from scipy.optimize import brentq, linprog
from scipy import sparse

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "outputs", "figures"); TAB = os.path.join(OUT, "outputs", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
rng = np.random.default_rng(20261012)
R = {}  # results collector

# ---------- plotting style (validated categorical palette, fixed order) ----------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.labelsize": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.dpi": 110,
})
def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)

YEARS = np.arange(2024, 2036)
T = YEARS - 2024

# =====================================================================
# M1  Scenario model
# =====================================================================
# Calibration anchors (sources in chapter): IEA Energy & AI (2025): 415 TWh in 2024,
# AI servers 15% of DC electricity, ~945 TWh 2030, ~1,200 TWh 2035, range 700-1,700.
# Ember: global grid intensity 473 g/kWh (2024), 458 g/kWh (2025, -2.7%/yr).
E0, AI_SHARE0 = 415.0, 0.15
PUE0 = 1.40          # energy-weighted fleet PUE (assumption: between hyperscale ~1.1 and survey mean 1.54-1.56)
CI0 = 473.0          # gCO2/kWh
WUE0 = 1.5           # L per kWh IT, direct on-site (fleet blend; survey averages 1.2-1.9)
WIND0 = 2.0          # L per kWh electricity, indirect (power-generation) water

def trajectory(g_ai1, g_ai2, g_c1, g_c2, pue30, pue35, d, wue35, wind0=WIND0, wue0=WUE0):
    """Return dict of yearly arrays for a parameter set (vectorised over leading dims)."""
    g_ai1, g_ai2, g_c1, g_c2, pue30, pue35, d, wue35, wind0, wue0 = map(
        lambda a: np.asarray(a, float)[..., None],
        (g_ai1, g_ai2, g_c1, g_c2, pue30, pue35, d, wue35, wind0, wue0))
    it_ai0 = E0 * AI_SHARE0 / PUE0
    it_c0 = E0 * (1 - AI_SHARE0) / PUE0
    t1 = np.minimum(T, 6); t2 = np.maximum(T - 6, 0)
    it_ai = it_ai0 * (1 + g_ai1) ** t1 * (1 + g_ai2) ** t2
    it_c = it_c0 * (1 + g_c1) ** t1 * (1 + g_c2) ** t2
    pue = np.where(T <= 6, PUE0 + (pue30 - PUE0) * T / 6, pue30 + (pue35 - pue30) * (T - 6) / 5)
    ci = CI0 * (1 - d) ** T
    wue = wue0 + (wue35 - wue0) * T / 11
    it = it_ai + it_c
    elec = it * pue
    co2 = elec * ci / 1000.0                      # Mt CO2
    wind = wind0 * (ci / CI0)                     # indirect water scales with thermal share
    water = (it * wue + elec * wind) / 1000.0     # TWh*L/kWh = 1e9 L ; /1000 -> km3 (1e12 L)
    water_direct = it * wue / 1000.0
    return dict(it_ai=it_ai, it_c=it_c, pue=pue * np.ones_like(it), ci=ci * np.ones_like(it),
                elec=elec, ai_elec=it_ai * pue, co2=co2, water=water, water_direct=water_direct)

# calibrate conventional growth so Base hits IEA's ~945 TWh in 2030
BASE = dict(g_ai1=0.30, g_ai2=0.10, g_c2=0.03, pue30=1.33, pue35=1.28, d=0.027, wue35=1.0)
def f_cal(gc):
    return trajectory(BASE["g_ai1"], BASE["g_ai2"], gc, BASE["g_c2"], BASE["pue30"],
                      BASE["pue35"], BASE["d"], BASE["wue35"])["elec"][..., 6].item() - 945
GC1 = brentq(f_cal, 0.0, 0.3)
BASE["g_c1"] = GC1

SCEN = {
    "Base":            BASE,
    "Lift-Off":        dict(g_ai1=0.36, g_ai2=0.16, g_c1=GC1 + 0.02, g_c2=0.05, pue30=1.34, pue35=1.30, d=0.020, wue35=1.2),
    "High Efficiency": dict(g_ai1=0.26, g_ai2=0.07, g_c1=GC1 - 0.03, g_c2=0.015, pue30=1.27, pue35=1.18, d=0.040, wue35=0.6),
    "Headwinds":       dict(g_ai1=0.20, g_ai2=0.05, g_c1=GC1 - 0.05, g_c2=0.010, pue30=1.36, pue35=1.33, d=0.020, wue35=1.2),
}
scen_out = {k: {kk: np.squeeze(v) for kk, v in trajectory(**p).items()} for k, p in SCEN.items()}

rows = []
for k, o in scen_out.items():
    for y in (2024, 2030, 2035):
        i = y - 2024
        rows.append(dict(Scenario=k, Year=y, Electricity_TWh=o["elec"][i], AI_share_pct=100 * o["ai_elec"][i] / o["elec"][i],
                         PUE=o["pue"][i], Grid_gCO2_kWh=o["ci"][i], CO2_Mt=o["co2"][i], Water_km3=o["water"][i],
                         Direct_water_km3=o["water_direct"][i]))
df_scen = pd.DataFrame(rows).round(2)
df_scen.to_csv(os.path.join(TAB, "T3_scenarios.csv"), index=False)
R["calibrated_g_c1"] = GC1
R["scenarios"] = df_scen.to_dict("records")

# parameter table
params = pd.DataFrame([
    ["Data-centre electricity 2024", "415 TWh", "IEA (2025)"],
    ["AI-server share of DC electricity 2024", "15 %", "IEA (2025)"],
    ["Energy-weighted fleet PUE 2024", "1.40", "Assumption bounded by hyperscale 1.10-1.15 and survey mean 1.54-1.56 (Uptime 2024, 2025)"],
    ["Grid carbon intensity 2024", "473 gCO2/kWh", "Ember (2025)"],
    ["Grid decarbonisation rate (base)", "2.7 %/yr", "Ember (2026): 2025 decline"],
    ["Direct WUE 2024 (fleet blend)", "1.5 L/kWh IT", "Range 1.2-1.9 in industry surveys"],
    ["Indirect water intensity of electricity", "2.0 L/kWh", "Global average used in literature"],
    ["AI IT growth 2024-30 / 2030-35 (base)", "30 % / 10 %", "IEA (2025): accelerated servers ~30 %/yr"],
    ["Conventional IT growth 2024-30 (base)", f"{100*GC1:.1f} %", "Calibrated to 945 TWh in 2030"],
    ["Conventional IT growth 2030-35 (base)", "3 %", "Assumption"],
    ["Fleet PUE 2030 / 2035 (base)", "1.33 / 1.28", "Assumption (new-build share rising)"],
], columns=["Parameter", "Value", "Basis"])
params.to_csv(os.path.join(TAB, "T2_parameters.csv"), index=False)

# ---------------------------------------------------------------------
# Figure 1: conceptual framework
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.0)
def box(x, y, w, h, title, lines, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=col + "22", ec=col, lw=1.6))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.55, "\n".join(lines), ha="center", va="top", fontsize=8, color=INK2, linespacing=1.35)
box(0.1, 3.0, 2.9, 1.85, "Demand drivers", ["Generative-AI training", "AI inference at scale", "Cloud, streaming, IoT", "Industry 4.0 data flows"], C[0])
box(3.55, 3.0, 2.9, 1.85, "Infrastructure layer", ["IT load (servers, storage, network)", "Cooling & power chain (PUE)", "Grid mix & siting", "Hardware refresh cycles"], C[6])
box(7.0, 3.0, 2.9, 1.85, "Environmental impacts", ["Electricity & grid stress", "Scope 2 CO2 (location/market)", "Direct + indirect water", "Embodied carbon & e-waste"], C[7])
box(1.6, 0.1, 6.8, 1.75, "Digital leadership levers (governance, strategy, culture)",
    ["Measure: hourly carbon, WUE, embodied-carbon disclosure", "Design: efficiency-first AI, liquid cooling, heat reuse",
     "Operate: carbon-aware scheduling, 24/7 clean energy procurement", "Govern: circularity, siting rules, board-level accountability"], C[2])
for x0, x1 in ((3.0, 3.55), (6.45, 7.0)):
    ax.annotate("", xy=(x1, 3.95), xytext=(x0, 3.95), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
for x in (1.55, 5.0):
    ax.annotate("", xy=(x, 2.98), xytext=(x, 1.88), arrowprops=dict(arrowstyle="-|>", color=C[2], lw=1.6))
ax.annotate("", xy=(7.6, 2.98), xytext=(7.6, 1.88), arrowprops=dict(arrowstyle="-|>", color=C[2], lw=1.6))
ax.annotate("", xy=(8.4, 1.88), xytext=(9.0, 2.98), arrowprops=dict(arrowstyle="-|>", color=C[7], lw=1.3, ls="--"))
ax.text(9.15, 2.35, "feedback:\ndisclosure,\nregulation,\nlicence to\noperate", fontsize=7.3, color=C[7], ha="left", va="center")
ax.set_xlim(0, 10.4)
save(fig, "fig1_framework.png")

# ---------------------------------------------------------------------
# Figure 2: empirical evidence (PUE stagnation; published projections)
# ---------------------------------------------------------------------
pue_hist = pd.DataFrame({"year": [2007, 2011, 2014, 2018, 2024, 2025], "pue": [2.50, 1.98, 1.65, 1.58, 1.56, 1.54]})
pue_hist.to_csv(os.path.join(TAB, "data_pue_history.csv"), index=False)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
a = axs[0]
a.plot(pue_hist.year, pue_hist.pue, "-o", color=C[0], ms=6)
for _, r in pue_hist.iterrows():
    a.annotate(f"{r.pue:.2f}", (r.year, r.pue), textcoords="offset points", xytext=(-9 if r.year == 2024 else (9 if r.year == 2025 else 0), 8), ha="center", fontsize=8, color=INK)
a.axhspan(1.10, 1.15, color=C[2], alpha=0.18); a.text(2008, 1.17, "Hyperscale band 1.10-1.15", fontsize=8, color=INK2)
a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); a.set_ylim(1.0, 2.75); a.set_title("(a) Survey-average PUE has plateaued since ~2018"); a.set_ylabel("PUE (total / IT energy)")
b = axs[1]
b.plot([2014, 2023], [58, 176], "-o", color=C[1], label="United States, LBNL (historical)")
b.fill_between([2023, 2028], [176, 325], [176, 580], color=C[1], alpha=0.18)
b.plot([2023, 2028], [176, 452.5], "--", color=C[1], lw=1.4)
b.plot([2024, 2030, 2035], [415, 945, 1200], "-o", color=C[0], label="World, IEA Base Case")
b.fill_between([2024, 2035], [415, 700], [415, 1700], color=C[0], alpha=0.12)
b.text(2033.2, 1500, "IEA 2035\nrange\n700-1,700", fontsize=7.5, color=INK2, ha="center")
b.text(2028.3, 330, "LBNL 2028:\n325-580", fontsize=7.5, color=INK2)
b.set_ylabel("TWh per year"); b.set_title("(b) Published demand trajectories"); b.legend(loc="upper left", fontsize=8)
fig.tight_layout(); save(fig, "fig2_evidence.png")

# ---------------------------------------------------------------------
# Figure 3: scenario results
# ---------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
o = scen_out["Base"]
a = axs[0]
a.bar(YEARS, o["it_c"] * o["pue"], color=C[0], width=0.8, label="Conventional workloads", edgecolor="white", lw=1)
a.bar(YEARS, o["ai_elec"], bottom=o["it_c"] * o["pue"], color=C[1], width=0.8, label="AI-accelerated servers", edgecolor="white", lw=1)
for y in (2024, 2030, 2035):
    i = y - 2024; a.text(y, o["elec"][i] + 25, f"{o['elec'][i]:.0f}", ha="center", fontsize=8, color=INK)
a.set_title("(a) Base case: facility electricity by workload"); a.set_ylabel("TWh per year"); a.legend(loc="upper left", fontsize=8)
a = axs[1]
for j, (k, oo) in enumerate(scen_out.items()):
    a.plot(YEARS, oo["co2"], color=C[[0, 7, 2, 6][j]], label=k)
    a.text(2035.25, oo["co2"][-1], f"{oo['co2'][-1]:.0f}", fontsize=8, va="center", color=INK)
a.set_title("(b) Operational CO$_2$ under four scenarios"); a.set_ylabel("Mt CO$_2$ per year"); a.legend(fontsize=8, loc="upper left")
a.set_xlim(2023.5, 2036.3)
fig.tight_layout(); save(fig, "fig3_scenarios.png")

# =====================================================================
# M2  Monte Carlo
# =====================================================================
N = 20000
def tri(lo, mode, hi, n=N): return rng.triangular(lo, mode, hi, n)
mc_par = dict(
    g_ai1=tri(0.18, 0.30, 0.40), g_ai2=tri(0.03, 0.10, 0.18), g_c1=tri(GC1 - 0.05, GC1, GC1 + 0.03),
    g_c2=tri(0.0, 0.03, 0.06), pue30=tri(1.25, 1.33, 1.40), pue35=None, d=tri(0.010, 0.027, 0.050),
    wue35=tri(0.4, 1.0, 1.6), wind0=tri(1.2, 2.0, 3.0), wue0=tri(1.2, 1.5, 1.9))
mc_par["pue35"] = mc_par["pue30"] - tri(0.0, 0.05, 0.12)
mc = trajectory(**mc_par)
def pct(x, q): return np.percentile(x, q, axis=0)
mc_rows = []
for var, unit in (("elec", "TWh"), ("co2", "Mt CO2"), ("water", "km3"), ("ai_elec", "TWh")):
    for y in (2030, 2035):
        i = y - 2024; x = mc[var][:, i]
        mc_rows.append(dict(Variable=var, Unit=unit, Year=y, Mean=x.mean(), P5=np.percentile(x, 5), P25=np.percentile(x, 25),
                            Median=np.median(x), P75=np.percentile(x, 75), P95=np.percentile(x, 95)))
df_mc = pd.DataFrame(mc_rows).round(1); df_mc.to_csv(os.path.join(TAB, "T4_montecarlo.csv"), index=False)
co2_2024 = mc["co2"][0, 0]
R["mc"] = df_mc.to_dict("records")
R["mc_prob_co2_2035_above_2024"] = float((mc["co2"][:, -1] > co2_2024).mean())
R["mc_prob_co2_2035_above_1p5x_2024"] = float((mc["co2"][:, -1] > 1.5 * co2_2024).mean())
R["mc_prob_elec_2035_above_1200"] = float((mc["elec"][:, -1] > 1200).mean())
R["co2_2024"] = float(co2_2024)
R["water_2024"] = float(np.squeeze(scen_out["Base"]["water"])[0])
R["water_direct_2024"] = float(np.squeeze(scen_out["Base"]["water_direct"])[0])
share_ind = 1 - mc["water_direct"][:, 0] / mc["water"][:, 0]
R["indirect_water_share_2024_median"] = float(np.median(share_ind))

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7))
for a, var, lab, col in zip(axs, ("elec", "co2", "water"), ("Electricity (TWh/yr)", "CO$_2$ (Mt/yr)", "Water consumption (km$^3$/yr)"), (C[0], C[7], C[2])):
    x = mc[var]
    a.fill_between(YEARS, pct(x, 5), pct(x, 95), color=col, alpha=0.15, lw=0, label="5-95 %")
    a.fill_between(YEARS, pct(x, 25), pct(x, 75), color=col, alpha=0.32, lw=0, label="25-75 %")
    a.plot(YEARS, pct(x, 50), color=col, label="Median")
    a.set_title(lab); a.set_xlim(2024, 2035)
axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"Monte Carlo uncertainty envelopes (N = {N:,})", fontsize=10.5, y=1.02)
fig.tight_layout(); save(fig, "fig4_montecarlo.png")

# =====================================================================
# M3  Sobol sensitivity (Saltelli 2010 S1, Jansen ST)
# =====================================================================
names = ["g_ai1", "g_ai2", "g_c1", "g_c2", "pue35", "d", "wue35", "wind0"]
labels = {"g_ai1": "AI growth 2024-30", "g_ai2": "AI growth 2030-35", "g_c1": "Conventional growth 2024-30",
          "g_c2": "Conventional growth 2030-35", "pue35": "Fleet PUE 2035", "d": "Grid decarbonisation rate",
          "wue35": "Direct WUE 2035", "wind0": "Indirect water intensity"}
bounds = {"g_ai1": (0.18, 0.40), "g_ai2": (0.03, 0.18), "g_c1": (GC1 - 0.05, GC1 + 0.03), "g_c2": (0.0, 0.06),
          "pue35": (1.12, 1.40), "d": (0.010, 0.050), "wue35": (0.4, 1.6), "wind0": (1.2, 3.0)}
NS = 16384; k = len(names)
A = rng.random((NS, k)); B = rng.random((NS, k))
def scale(U): return {n: bounds[n][0] + U[:, j] * (bounds[n][1] - bounds[n][0]) for j, n in enumerate(names)}
def model(U):
    p = scale(U); pue30 = np.clip(p["pue35"] + 0.05, None, 1.40)
    o = trajectory(p["g_ai1"], p["g_ai2"], p["g_c1"], p["g_c2"], pue30, p["pue35"], p["d"], p["wue35"], wind0=p["wind0"])
    return o["co2"][:, -1], o["water"][:, -1], o["elec"][:, -1]
fA, fB = model(A), model(B)
sob = {}
for oi, out in enumerate(("CO2_2035", "Water_2035", "Elec_2035")):
    yA, yB = fA[oi], fB[oi]; V = np.var(np.r_[yA, yB])
    S1, ST = [], []
    for j in range(k):
        ABj = A.copy(); ABj[:, j] = B[:, j]; yAB = model(ABj)[oi]
        S1.append(np.mean(yB * (yAB - yA)) / V)
        ST.append(0.5 * np.mean((yA - yAB) ** 2) / V)
    sob[out] = (np.array(S1), np.array(ST))
df_sob = pd.DataFrame({"Parameter": [labels[n] for n in names]})
for out, (s1, st) in sob.items():
    df_sob[f"{out}_S1"] = s1.round(3); df_sob[f"{out}_ST"] = st.round(3)
df_sob.to_csv(os.path.join(TAB, "T5_sobol.csv"), index=False)
R["sobol"] = df_sob.to_dict("records")

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
ypos = np.arange(k)
for a, out, ttl in zip(axs, ("CO2_2035", "Water_2035"), ("(a) Operational CO$_2$ in 2035", "(b) Total water consumption in 2035")):
    s1, st = sob[out]; order = np.argsort(sob["CO2_2035"][1])
    a.barh(ypos - 0.19, st[order], height=0.36, color=C[0], label="Total-order $S_T$")
    a.barh(ypos + 0.19, s1[order], height=0.36, color=C[1], label="First-order $S_1$")
    a.set_yticks(ypos); a.set_yticklabels([labels[names[i]] for i in order]); a.set_title(ttl); a.set_xlabel("Sobol index (share of output variance)")
    a.set_xlim(0, max(0.6, st.max() * 1.15)); a.grid(axis="y", visible=False)
axs[0].legend(fontsize=8, loc="lower right")
fig.tight_layout(); save(fig, "fig5_sobol.png")

# =====================================================================
# M4  Carbon-aware scheduling LP
# =====================================================================
H = 168; h = np.arange(H); hod = h % 24; day = h // 24
solar = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None)
cloud = np.array([1.0, 0.85, 0.55, 0.95, 1.0, 0.7, 0.9])[day]           # weather variation
evening = np.exp(-0.5 * ((hod - 20) / 1.6) ** 2)
noise = np.zeros(H); e = rng.normal(0, 12, H)
for t in range(1, H): noise[t] = 0.7 * noise[t - 1] + e[t]
CIh = 760 - 380 * solar * cloud + 55 * evening + noise                 # coal-heavy grid with rising solar
R["sched_CI_mean"] = float(CIh.mean()); R["sched_CI_min"] = float(CIh.min()); R["sched_CI_max"] = float(CIh.max())
CAP_IT = 100.0                                                         # MW
inflex_shape = 1 + 0.12 * np.sin(2 * np.pi * (hod - 15) / 24)          # inference follows users

def schedule(flex_frac, headroom, W):
    """Return (baseline_t, optimised_t, x-profile) for a facility whose average IT load = CAP/headroom."""
    # headroom = IT capacity / peak carbon-agnostic load (so the baseline is always feasible)
    peak_factor = ((1 - flex_frac) * inflex_shape + flex_frac).max()
    mean_load = CAP_IT / (headroom * peak_factor)
    inflex = (1 - flex_frac) * mean_load * inflex_shape
    flex_arr = np.full(H, flex_frac * mean_load)                      # batch jobs arrive uniformly
    base_load = inflex + flex_arr
    if flex_frac == 0: return base_load, base_load
    idx = [(t, s) for t in range(H) for s in range(t, min(t + W, H))]
    nv = len(idx); cost = np.array([CIh[s] for t, s in idx])
    rows_eq = np.array([t for t, s in idx]); rows_ub = np.array([s for t, s in idx])
    Aeq = sparse.csr_matrix((np.ones(nv), (rows_eq, np.arange(nv))), shape=(H, nv))
    Aub = sparse.csr_matrix((np.ones(nv), (rows_ub, np.arange(nv))), shape=(H, nv))
    res = linprog(cost, A_ub=Aub, b_ub=np.maximum(CAP_IT - inflex, 0), A_eq=Aeq, b_eq=flex_arr, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    x = np.zeros(H)
    np.add.at(x, rows_ub, res.x)
    return base_load, inflex + x

PUE_S = 1.2
def emis(load): return float((load * PUE_S * CIh).sum() / 1e6)        # MW*h*g/kWh -> t ; /1e6 -> kt? keep relative
flex_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
head_grid = [1.05, 1.15, 1.25, 1.35, 1.5, 1.75]
red = np.zeros((len(flex_grid), len(head_grid)))
for i, f in enumerate(flex_grid):
    for j, hr in enumerate(head_grid):
        b, opt = schedule(f, hr, 24)
        red[i, j] = 100 * (1 - emis(opt) / emis(b))
df_red = pd.DataFrame(red, index=[f"{int(100*f)}% flexible" for f in flex_grid], columns=[f"headroom {hr:.2f}" for hr in head_grid]).round(2)
df_red.to_csv(os.path.join(TAB, "T6_scheduling_grid.csv"))
win_res = []
for W in (1, 4, 8, 12, 24, 48):
    b, opt = schedule(0.3, 1.35, W); win_res.append((W, 100 * (1 - emis(opt) / emis(b))))
df_win = pd.DataFrame(win_res, columns=["Deferral window (h)", "CO2 reduction (%)"]).round(2)
df_win.to_csv(os.path.join(TAB, "T6b_scheduling_window.csv"), index=False)
b30, o30 = schedule(0.3, 1.35, 24)
R["sched_ref_reduction"] = float(df_win.loc[df_win["Deferral window (h)"] == 24, "CO2 reduction (%)"].iloc[0])
R["sched_grid"] = df_red.to_dict(); R["sched_window"] = win_res
R["sched_ref_abs_t_week"] = float((emis(b30) - emis(o30)) * 1e3)      # emis() in kt? -> compute tonnes
# emis(): MW * h * g/kWh = MWh*g/kWh = 1000 kWh*g/kWh = 1000 g = 1 kg ; so sum is kg ; /1e6 -> kt
R["sched_ref_base_kt_week"] = emis(b30); R["sched_ref_opt_kt_week"] = emis(o30)

fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)
a = fig.add_subplot(gs[0, :])
a.plot(h, b30, color=INK2, lw=1.4, label="IT load, carbon-agnostic (MW)")
a.plot(h, o30, color=C[0], lw=2, label="IT load, carbon-aware (MW)")
a.axhline(CAP_IT, color=C[7], lw=1, ls="--"); a.text(2, CAP_IT + 2, "IT capacity cap", color=C[7], fontsize=8)
a.set_ylabel("IT load (MW)"); a.set_ylim(0, 112); a.set_xlim(0, 167); a.set_xticks(range(0, 169, 24)); a.set_xticklabels([f"Day {d+1}" for d in range(7)] + [""])
a.set_title("(a) One week: 30 % flexible batch AI load, 24 h deferral window, headroom 1.35 (grid intensity shaded)")
a2 = a.inset_axes([0, -0.0, 1, 0.22], sharex=a); a2.axis("off")
for t in range(H):
    a.axvspan(t, t + 1, ymin=0, ymax=0.06, color=plt.cm.YlOrBr((CIh[t] - 300) / 550), lw=0)
a2.remove()
a.legend(fontsize=8, loc="upper right", ncol=2)
a = fig.add_subplot(gs[1, 0])
im = a.imshow(red, cmap="Blues", aspect="auto", origin="lower")
a.set_xticks(range(len(head_grid))); a.set_xticklabels([f"{x:.2f}" for x in head_grid]); a.set_yticks(range(len(flex_grid))); a.set_yticklabels([f"{int(100*f)}%" for f in flex_grid])
for i in range(len(flex_grid)):
    for j in range(len(head_grid)):
        a.text(j, i, f"{red[i,j]:.1f}", ha="center", va="center", fontsize=7.5, color="white" if red[i, j] > red.max() * 0.55 else INK)
a.set_xlabel("Capacity headroom (cap / peak IT load)"); a.set_ylabel("Flexible share of IT load"); a.set_title("(b) CO$_2$ reduction (%) vs flexibility and headroom"); a.grid(False)
a = fig.add_subplot(gs[1, 1])
a.plot(df_win.iloc[:, 0], df_win.iloc[:, 1], "-o", color=C[0], ms=7)
for x_, y_ in win_res: a.annotate(f"{y_:.1f}%", (x_, y_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
a.set_xscale("log", base=2); a.set_xticks([1, 4, 8, 12, 24, 48]); a.set_xticklabels(["1", "4", "8", "12", "24", "48"])
a.set_xlabel("Maximum deferral window (hours)"); a.set_ylabel("CO$_2$ reduction (%)"); a.set_title("(c) Value of temporal flexibility (30 %, headroom 1.35)")
a.set_ylim(0, max(df_win.iloc[:, 1]) * 1.25)
save(fig, "fig6_scheduling.png")

# =====================================================================
# M5  Hardware refresh lifecycle
# =====================================================================
EMB = 25.0     # t CO2e embodied per 8-accelerator server (assumption; literature range ~15-40)
P_KW = 10.0; UTIL = 0.7; PUE_L = 1.2
MASS = 120.0   # kg per server (assumption)
R_EFF = 0.25   # yearly gain in work per kWh for new hardware
Q_PERF = 0.35  # yearly gain in work per server for new hardware
HZN = 12; FLEET0 = 100
def lifecycle(L, ci0, dgrid=0.027, r_eff=None):
    r_eff = R_EFF if r_eff is None else r_eff
    tot_op = tot_emb = waste = 0.0
    for t in range(HZN):
        vintage = (t // L) * L
        servers = FLEET0 / (1 + Q_PERF) ** vintage
        energy_mwh = FLEET0 * P_KW * UTIL * 8760 * PUE_L / 1000 / (1 + r_eff) ** vintage
        tot_op += energy_mwh * ci0 * (1 - dgrid) ** t / 1000           # MWh * g/kWh = kg ; /1000 -> t
        if t == vintage:
            tot_emb += servers * EMB * min(L, HZN - vintage) / L   # amortise the last vintage over years used
            if t > 0: waste += FLEET0 / (1 + Q_PERF) ** (vintage - L) * MASS / 1000
    waste += servers * MASS / 1000                                         # end-of-horizon retirement
    return tot_op, tot_emb, waste
Ls = [1, 2, 3, 4, 5, 6]; cis = [50, 150, 300, 450, 600, 750]
lc = []
for c in cis:
    for L in Ls:
        op, em, w = lifecycle(L, c); lc.append(dict(Grid_gCO2_kWh=c, Lifetime_yr=L, Operational_t=op, Embodied_t=em, Total_t=op + em, Ewaste_t=w))
df_lc = pd.DataFrame(lc).round(1); df_lc.to_csv(os.path.join(TAB, "T7_lifecycle.csv"), index=False)
opt = df_lc.loc[df_lc.groupby("Grid_gCO2_kWh").Total_t.idxmin()][["Grid_gCO2_kWh", "Lifetime_yr", "Total_t", "Ewaste_t"]]
R["lifecycle_optimal"] = opt.to_dict("records")
# continuous map of optimal lifetime
ci_fine = np.arange(25, 801, 25); Lfine = np.arange(1, 8)
optLs = {re: [int(Lfine[np.argmin([sum(lifecycle(L, c, r_eff=re)[:2]) for L in Lfine])]) for c in ci_fine] for re in (0.15, 0.25, 0.35)}
optL = optLs[0.25]
R["lifecycle_optL_curve"] = {str(re): list(zip(ci_fine.tolist(), v)) for re, v in optLs.items()}
pd.DataFrame({"Grid_gCO2_kWh": ci_fine, **{f"OptL_eff{int(re*100)}": v for re, v in optLs.items()}}).to_csv(os.path.join(TAB, "T7b_optimal_lifetime.csv"), index=False)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
a = axs[0]
for j, c in enumerate([50, 450, 750]):
    d_ = df_lc[df_lc.Grid_gCO2_kWh == c]
    a.plot(d_.Lifetime_yr, d_.Total_t / 1000, "-o", color=C[[2, 0, 7][j]], label=f"{c} gCO$_2$/kWh grid", ms=6)
    m = d_.loc[d_.Total_t.idxmin()]; a.plot(m.Lifetime_yr, m.Total_t / 1000, marker="*", ms=15, color=C[[2, 0, 7][j]], mec="white")
a.set_xlabel("Hardware refresh interval (years)"); a.set_ylabel("12-year life-cycle CO$_2$e (kt)"); a.set_title("(a) Total carbon vs refresh interval (star = optimum)")
a.legend(fontsize=8)
a = axs[1]
for j, re in enumerate((0.15, 0.25, 0.35)):
    a.step(ci_fine, np.array(optLs[re]) + (j - 1) * 0.06, where="mid", color=C[[2, 6, 1][j]], label=f"new-hardware efficiency gain {int(re*100)} %/yr")
a.legend(fontsize=7.5); a.set_xlabel("Initial grid carbon intensity (gCO$_2$/kWh)"); a.set_ylabel("Carbon-optimal refresh interval (yr)")
a.set_title("(b) Cleaner grids favour longer hardware lives"); a.set_yticks(range(1, 8))
fig.tight_layout(); save(fig, "fig7_lifecycle.png")

# =====================================================================
# M6a  Abatement waterfall, 2035 base case
# =====================================================================
ob = {k_: np.squeeze(v)[-1] for k_, v in scen_out["Base"].items()}
it_ai, it_c, pue, ci = ob["it_ai"], ob["it_c"], ob["pue"], ob["ci"]
steps = [("2035 Base-case emissions", it_ai, it_c, pue, ci)]
pue2 = 1.15; steps.append(("Facility efficiency\n(PUE 1.28 -> 1.15)", it_ai, it_c, pue2, ci))
it_ai2 = it_ai * 0.75; it_c2 = it_c * 0.90; steps.append(("Efficient AI & IT\n(-25 % AI, -10 % other)", it_ai2, it_c2, pue2, ci))
sched_f = R["sched_ref_reduction"] / 100
def em(ia, ic, p, c, ai_sched=0.0): return (ia * p * c * (1 - ai_sched) + ic * p * c) / 1000
vals = [em(*s[1:]) for s in steps]
vals.append(em(it_ai2, it_c2, pue2, ci, sched_f)); steps.append(("Carbon-aware\nscheduling (AI load)",))
cfe = 0.40; vals.append(vals[-1] * (1 - cfe)); steps.append(("24/7 clean-energy\nmatching (+40 %)",))
labels_w = [s[0] for s in steps]
deltas = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
df_w = pd.DataFrame({"Step": [l.replace("\n", " ") for l in labels_w] + ["Residual 2035"], "Change_Mt": deltas + [vals[-1]],
                     "Level_after_Mt": vals + [vals[-1]]}).round(1)
df_w.to_csv(os.path.join(TAB, "T8_abatement.csv"), index=False)
R["waterfall"] = df_w.to_dict("records")
fig, a = plt.subplots(figsize=(10.5, 4.6))
xs = np.arange(len(vals) + 1); cur = 0
for i, d_ in enumerate(deltas):
    if i == 0: a.bar(i, d_, color=C[7], width=0.62); cur = d_; a.text(i, d_ + 8, f"{d_:.0f}", ha="center", fontsize=8.5)
    else:
        a.bar(i, d_, bottom=cur, color=C[2], width=0.62); a.text(i, cur + d_ - 8, f"{d_:.0f}", ha="center", va="top", fontsize=8.5); cur += d_
a.bar(len(vals), vals[-1], color=C[0], width=0.62); a.text(len(vals), vals[-1] + 8, f"{vals[-1]:.0f}", ha="center", fontsize=8.5)
a.axhline(R["co2_2024"], color=INK2, ls="--", lw=1); a.text(0.42, R["co2_2024"] + 6, f"2024 level ({R['co2_2024']:.0f} Mt)", fontsize=8, color=INK2, ha="left")
a.set_ylim(0, vals[0] * 1.12)
a.set_xticks(xs); a.set_xticklabels([l.replace("2035 Base-case emissions", "2035 Base\ncase") for l in labels_w] + ["Residual\n2035"], fontsize=7.8); a.set_ylabel("Mt CO$_2$ per year")
a.set_title("Sequential abatement of 2035 Base-case data-centre emissions"); a.grid(axis="x", visible=False)
save(fig, "fig8_waterfall.png")

# =====================================================================
# M6b  TOPSIS ranking of leadership levers with weight uncertainty
# =====================================================================
alts = ["Liquid cooling & PUE retrofit", "24/7 carbon-free energy PPAs", "Carbon-aware workload scheduling",
        "Model efficiency (distil, quantise, right-size)", "Hardware life extension & circularity",
        "Waste-heat reuse", "Low-carbon, low-water-stress siting"]
crit = ["Abatement potential", "Capital cost", "Time to impact", "Water co-benefit", "Governance complexity"]
benefit = np.array([1, 0, 0, 1, 0], bool)
# 1-9 scores: author's structured judgement synthesised from the evidence reviewed in Sections 2-4
X = np.array([
    [5, 7, 5, 6, 3],
    [9, 8, 6, 5, 6],
    [4, 2, 2, 3, 4],
    [8, 2, 2, 5, 3],
    [5, 3, 3, 3, 5],
    [3, 6, 6, 2, 7],
    [7, 8, 8, 9, 7],
], float)
pd.DataFrame(X, index=alts, columns=crit).to_csv(os.path.join(TAB, "T9a_topsis_scores.csv"))
def topsis(X, w):
    Nm = X / np.sqrt((X ** 2).sum(0)); V = Nm * w
    best = np.where(benefit, V.max(0), V.min(0)); worst = np.where(benefit, V.min(0), V.max(0))
    dp = np.sqrt(((V - best) ** 2).sum(1)); dm = np.sqrt(((V - worst) ** 2).sum(1))
    return dm / (dp + dm)
w0 = np.array([0.35, 0.20, 0.20, 0.10, 0.15])
cc0 = topsis(X, w0)
W = rng.dirichlet(w0 * 40, 10000)
ranks = np.array([(-topsis(X, w)).argsort().argsort() + 1 for w in W])
df_t = pd.DataFrame({"Lever": alts, "Closeness_base": cc0.round(3), "Rank_base": (-cc0).argsort().argsort() + 1,
                     "P_rank1": (ranks == 1).mean(0).round(3), "P_top3": (ranks <= 3).mean(0).round(3),
                     "Median_rank": np.median(ranks, 0)}).sort_values("Rank_base")
df_t.to_csv(os.path.join(TAB, "T9_topsis.csv"), index=False)
R["topsis"] = df_t.to_dict("records")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
a = axs[0]; o_ = df_t.iloc[::-1]
a.barh(o_.Lever, o_.Closeness_base, color=C[0], height=0.6)
for y_, v in enumerate(o_.Closeness_base): a.text(v + 0.01, y_, f"{v:.2f}", va="center", fontsize=8)
a.set_xlim(0, 1); a.set_xlabel("TOPSIS closeness coefficient"); a.set_title("(a) Ranking at base weights"); a.grid(axis="y", visible=False)
a = axs[1]
rank_dist = np.array([[(ranks[:, alts.index(l)] == r).mean() for r in range(1, 8)] for l in o_.Lever])
left = np.zeros(len(o_)); seq = plt.cm.Blues(np.linspace(0.95, 0.15, 7))
for r in range(7):
    a.barh(range(len(o_)), rank_dist[:, r], left=left, color=seq[r], height=0.6, edgecolor="white", lw=1, label=f"Rank {r+1}")
    left += rank_dist[:, r]
a.set_yticks(range(len(o_))); a.set_yticklabels([]); a.set_xlim(0, 1); a.set_xlabel("Share of 10,000 weight draws")
a.set_title("(b) Rank stability under weight uncertainty"); a.legend(fontsize=7, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), handlelength=1)
a.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "fig9_topsis.png")

json.dump(R, open(os.path.join(OUT, "outputs", "results.json"), "w"), indent=1, default=float)
print("calibrated g_c1:", round(GC1, 4))
print(df_scen.to_string())
print(df_mc.to_string())
print(df_sob.to_string())
print(df_red.to_string()); print(df_win.to_string())
print(opt.to_string()); print(R["lifecycle_optL_curve"])
print(df_w.to_string()); print(df_t.to_string())
print({k_: R[k_] for k_ in ["mc_prob_co2_2035_above_2024", "mc_prob_co2_2035_above_1p5x_2024", "mc_prob_elec_2035_above_1200",
                             "co2_2024", "water_2024", "water_direct_2024", "indirect_water_share_2024_median",
                             "sched_CI_mean", "sched_CI_min", "sched_CI_max", "sched_ref_base_kt_week", "sched_ref_opt_kt_week"]})
