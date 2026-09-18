# Tail-Fragility, Barbell Portfolios, and Antifragility — Master's Thesis

**Title:** *Empirical Analysis of Fragility, Robustness, and Antifragility in Investment Portfolios*
**Author:** Gabriele Nicolasi
**Supervisor:** Prof. Marzia De Donno
**Università Cattolica del Sacro Cuore**, Faculty of Banking, Finance and Insurance Sciences, Milan — A.Y. 2024/2025

## Abstract

This thesis operationalizes Nassim Taleb's concepts of *fragility*, *robustness*, and *antifragility* as empirical, distribution-free measures of tail sensitivity, and tests them on real market data. Rather than assuming a theoretical return distribution, the framework builds tail-sensitivity metrics directly from the **empirical** distribution of returns — a left-tail measure `V` (downside) and a right-tail measure `W` (upside) — and studies how they respond to shocks in the tail threshold via a sensitivity coefficient, **Vega**. The framework is applied to three portfolio strategies:

1. **Passive** — unhedged exposure to the S&P 500 index (weekly log-returns, 1950–2025, n = 3,925)
2. **Hedged** — the same index protected by a protective put (97% strike)
3. **Barbell** — 85–95% allocated to a risk-free asset, with the remainder split across a basket of 15 high-volatility assets (large-cap tech, crypto, small/meme-cap equities, 2020–2025)

## Key results

| Strategy | Left-tail sensitivity (Vega) at K ≈ 5th percentile | Interpretation |
|---|---|---|
| Passive (unhedged S&P 500) | **0.836** | Most fragile: full exposure to tail losses |
| Barbell (85% RF / 15% risky → 95% RF / 5% risky) | **0.614 → 0.374** | Robust: sensitivity scales down with the risk-free share, but never reaches zero |
| Hedged (S&P 500 + protective put) | **0.000** | Antifragile: the put fully zeroes out sensitivity beyond its strike |

Under stress-testing (12 simulated shocks of −5% to −30% appended to the historical series), the ranking holds: the unhedged portfolio's Vega rises to **1.028**, while the hedged portfolio remains at **0**. On the upside (`W`), the hedged strategy is the only one able to both limit downside *and* partially retain upside exposure — under Taleb's own definition, it is the only strategy that qualifies as antifragile; the barbell only qualifies as robust, since it caps upside along with downside.

## Repository contents

| File | Description |
|---|---|
| `Thesis_EN.pdf` | Full thesis, English version |
| `Tesi_ita.pdf` | Full thesis, original Italian version (as submitted) |
| `Master_Thesis_Code_EN.ipynb` | Jupyter notebook (English), full pipeline: data download, tail-sensitivity functions, protective-put and stress-test construction, barbell portfolio optimization (`cvxpy`), and all figures in the thesis |
| `Master's_thesis_code.py` | Original Python script version of the analysis |

## Methodology in brief

For a return threshold `Ω` and semi-deviation `s⁻` (mean shortfall below `Ω`), the framework rescales the empirical distribution by `λ = s⁻ / s⁻_baseline`, computes a truncated tail mass `ξ(K, s⁻)`, and defines **Vega** as the sensitivity of `ξ` to `s⁻` — a non-parametric measure of tail risk that assumes no particular theoretical distribution or tail shape. The same construction applied to the right tail (threshold `L`) yields **W**, the upside-sensitivity analogue. Barbell portfolios are built via convex optimization (`cvxpy`: minimum-variance, equal-weight, high-risk, and target-return variants) over a 15-asset risky basket plus a risk-free leg.

## Contact

**Gabriele Nicolasi** — [gabrielenicolasi.01@gmail.com](mailto:gabrielenicolasi.01@gmail.com) · [LinkedIn](https://www.linkedin.com/in/gabriele-nicolasi-b35259236)
