# Data-Centre Footprint and Digital Leadership for Sustainable AI

Code and data for the book chapter:

> **Powering Intelligence, Pricing the Planet: A Computational Assessment of the Environmental Footprint of Data Centres and the Digital-Leadership Agenda for Sustainable AI.** In *De Gruyter Handbook on Digital Leadership: Artificial Intelligence and Sustainability* (Part IV, Theme 20). Forthcoming.

One Python script reproduces every figure, table and number reported in the chapter. The random seed is fixed, so results are identical on every run.

## Quick start

```bash
pip install -r requirements.txt
python analysis.py
```

The script runs in about 10 seconds and writes everything to `outputs/`.

## Repository contents

| Path | Contents |
|---|---|
| `analysis.py` | The full analysis: six linked models (M1–M6) |
| `requirements.txt` | Python packages (NumPy, SciPy, pandas, Matplotlib) |
| `data/input_sources.csv` | The published anchor values used to calibrate the models, with their sources and links |
| `outputs/figures/` | The nine figures (300 dpi PNG) |
| `outputs/tables/` | Result tables (CSV) |
| `outputs/results.json` | Key numbers quoted in the chapter text |

## The six models

| Model | What it does | Section of `analysis.py` |
|---|---|---|
| M1 | Scenario model of data-centre electricity, CO₂ and water to 2035, calibrated to IEA (2025) projections; four scenarios | `M1` |
| M2 | Monte Carlo uncertainty propagation (N = 20,000) | `M2` |
| M3 | Variance-based (Sobol) sensitivity analysis, Saltelli (2010) and Jansen (1999) estimators | `M3` |
| M4 | Carbon-aware workload scheduling as a linear programme (HiGHS solver) | `M4`, function `schedule()` |
| M5 | Hardware-refresh life-cycle model of operational vs embodied carbon and e-waste | `M5`, function `lifecycle()` |
| M6 | Sequential abatement pathway and TOPSIS multi-criteria ranking of leadership levers with weight uncertainty | `M6a`, `M6b`, function `topsis()` |

## Figure map (chapter numbering)

| Chapter figure | File |
|---|---|
| Figure 1. Survey-average PUE and published electricity trajectories | `fig2_evidence.png` |
| Figure 2. Conceptual framework | `fig1_framework.png` |
| Figure 3. Base-case electricity and CO₂ under four scenarios | `fig3_scenarios.png` |
| Figure 4. Monte Carlo uncertainty envelopes | `fig4_montecarlo.png` |
| Figure 5. Sobol sensitivity indices | `fig5_sobol.png` |
| Figure 6. Carbon-aware scheduling results | `fig6_scheduling.png` |
| Figure 7. Hardware-refresh life-cycle results | `fig7_lifecycle.png` |
| Figure 8. Abatement waterfall to 2035 | `fig8_waterfall.png` |
| Figure 9. TOPSIS ranking and rank stability | `fig9_topsis.png` |

## Assumptions

Where the literature gives no single value, the model uses stated assumptions, listed in the chapter's Table 2. The main ones are an energy-weighted fleet PUE of 1.40 in 2024, 25 tCO₂e embodied carbon per eight-accelerator server, and an indirect water intensity of electricity of 2.0 L/kWh. The scheduling model uses a stylised one-week grid-intensity profile, not measured data. The TOPSIS scores (matrix `X` in `analysis.py`) are the author's structured judgement. The sensitivity analyses show how much each assumption matters; users are encouraged to substitute their own values.

## Licence

Code: MIT licence (see `LICENSE`). Figures, tables and the data file: Creative Commons Attribution 4.0 (CC BY 4.0). The anchor values in `data/input_sources.csv` remain the work of their original publishers and should be cited to them.

## How to cite

See `CITATION.cff`, or cite the book chapter once published.

## Development note

The code was developed with the assistance of an AI coding tool. It has been run end to end and reproduces the chapter's results; independent review of the modelling assumptions is welcome via GitHub issues.
