# GTA VI Trophy Rarity Forecast

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/888chunfeng168/gta6-trophy-rarity-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/888chunfeng168/gta6-trophy-rarity-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

An entry for the [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): a program that forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, from the house's recording of the trophy page up to the case's checkpoint.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every case file in the input directory and writes one forecast per case:

1. **Distinct page states.** Exophase only moves a game's counts when it refreshes, so runs of identical reads collapse into one state.
2. **Recent-window fits in log-time.** For each trophy, the logit of its earn rate is regressed on ln(1 + hours since release) over the last 36 hours of states, anchored at the last observed rate, with slopes shrunk toward the tier median.
3. **A saturating adoption curve.** Each trophy also gets a fit of r(t) = A * (1 - exp(-(t+2)/s)) over a grid of time constants; the two views are averaged by their window fit, so a rate that has levelled off is not extrapolated to 100%.
4. **A refresh prior for flat histories.** When no refresh was ever seen, the next one is expected to multiply the tracked base several-fold (638 -> 3,868 players on launch night), diluting every rate.
5. **Honest spread.** Quantile widths combine fit residuals, slope uncertainty over the extrapolation distance, disagreement between the two views, and a term growing with the horizon.
6. **Edge cases.** A trophy the whole tracked base already holds is forecast to stay near 100% with a long lower tail, never a point mass at 100. A trophy at 0% throughout its history stays near 0% with a small spread. Reads that lack a trophy (null rates) are skipped for that trophy only, and a flat history over a long horizon falls back to persistence with a wide spread instead of the short-horizon refresh prior.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local validation

Against the kit's practice cases, scored on the answer totals the kit documents (3,868 players, 4,276 trophies earned at the 04:30Z refresh), the forecaster scores a mean pinball loss of about 0.19 versus about 1.41 for the house persistence baseline. On synthetic final-like cases (saturating, still-rising and dilution-dominated regimes, scored exactly), it beats the baseline in every regime:

| case | this entry | house baseline |
| --- | --- | --- |
| practice p00 / p01 (reconstructed answer) | 0.19 | 1.41 |
| synthetic, saturating (3 seeds) | 0.15 – 0.17 | 0.25 – 0.26 |
| synthetic, still rising (2 seeds) | 0.13 – 0.14 | 0.51 – 0.52 |
| synthetic, dilution-dominated (2 seeds) | 0.12 – 0.13 | 0.23 |

Runs are deterministic (two runs on the same case write identical bytes) and take about 0.05 s per case locally, far inside the 8 s cap. A stress case with null rates, an always-zero trophy and 357 reads also validates cleanly.
