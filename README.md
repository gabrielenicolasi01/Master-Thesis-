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

## A note on "Vega"

The sensitivity coefficient used throughout this work is called **Vega**, following Nassim Taleb's own tail-risk terminology (*"tail Vega sensitivity"*, in his fragility/antifragility framework) — it is **not** the volatility-Vega of a single option contract, and the reuse of the name is not a coincidence: the thesis's **Fragility Transfer Theorem** (Eq. 27–30 below) proves that this coefficient is mathematically equivalent to a ratio of the Vegas of a portfolio of European put options replicating the tail (a barrier put vs. an at-the-money put). See Chapter 2 of the thesis for the full derivation.

## Key formulas

Given a random variable `X` with pdf `f_λ` (parameter `λ`) and reference threshold `Ω`, the **left absolute semi-deviation** is:

$$s^-(\lambda) = \int_{-\infty}^{\Omega} (\Omega - x)\, f_\lambda(x)\, dx$$

For a stress level `K < Ω`, the **truncated tail mass** (expected damage below `K`) is:

$$\xi(K, s^-) = \int_{-\infty}^{K} (\Omega - x)\, f_{\lambda(s^-)}(x)\, dx$$

The **(left-tail) Vega sensitivity** is the derivative of `ξ` with respect to `s⁻` — the sensitivity of the tail damage to an error in estimating the semi-deviation:

$$V(X, f_\lambda, K, s^-) = \frac{\partial \xi}{\partial s^-}(K, s^-) = \left(\int_{-\infty}^{K} (\Omega - x)\, \frac{\partial f_\lambda}{\partial \lambda}(x)\, dx\right)\left(\frac{ds^-}{d\lambda}\right)^{-1}$$

computed in the notebook via finite differences: $V = \frac{1}{2\Delta s}\big(\xi(K, s^- + \Delta s) - \xi(K, s^- - \Delta s)\big)$.

The right-tail analogue (**upside sensitivity, W**) mirrors this construction above the threshold, over an interval `[L, H]`:

$$s^+(\lambda) = \int_{\Omega}^{+\infty} (x - \Omega)\, f_\lambda(x)\, dx, \qquad W(X, f_\lambda, L, H, s^+) = \frac{\partial \xi^+}{\partial s^+}(L, H, s^+)$$

A portfolio is **fragile** if `V` is large (small errors in the tail estimate translate into large damage), **robust** if `V` stays below an acceptable bound `b` across the stress range, and **antifragile** if it is robust on the left (`V` small/zero) *and* still exposed on the right (`W > 0`) — i.e. it benefits from upside surprises while being shielded from downside ones.

## Methodology in brief

For a return threshold `Ω` and semi-deviation `s⁻` (mean shortfall below `Ω`), the framework rescales the empirical distribution by `λ = s⁻ / s⁻_baseline`, computes the truncated tail mass `ξ(K, s⁻)` above, and defines **Vega** as its sensitivity to `s⁻` — a non-parametric measure of tail risk that assumes no particular theoretical distribution or tail shape. The same construction applied to the right tail (threshold `L`) yields **W**, the upside-sensitivity analogue. Barbell portfolios are built via convex optimization (`cvxpy`: minimum-variance, equal-weight, high-risk, and target-return variants) over a 15-asset risky basket plus a risk-free leg.

**Hedge design.** The protective put used for the "hedged" portfolio is deliberately cheap: a **1-week-duration, far out-of-the-money put** (strike at 97% of the price, ≈3% OTM), priced at a premium of only **≈0.02% of the underlying price per period**. This is why the resulting Vega = 0 is not an unrealistically "free" result — it reflects a minimal, short-dated insurance position, not a heavily-bought, expensive hedge. The small but nonzero drop in the upside sensitivity `W` (0.014347 → 0.013719) is exactly the (small) price paid for that insurance.

## Contact

**Gabriele Nicolasi** — [gabrielenicolasi.01@gmail.com](mailto:gabrielenicolasi.01@gmail.com) · [LinkedIn](https://www.linkedin.com/in/gabriele-nicolasi-b35259236)
