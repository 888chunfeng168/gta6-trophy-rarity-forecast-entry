"""Forecast GTA VI trophy earn rates on Exophase at a case's target time.

Reads every case file (<id>.in) in the input directory and writes one forecast
file (<id>.out) per case: five quantiles (0.05/0.25/0.5/0.75/0.95) of each
trophy's earn rate, in percent, at the case's target.

How it forecasts (standard library only, no network, deterministic):

1. Collapse the case's history into its distinct page states.  Exophase only
   moves a game's counts when it refreshes, so long runs of identical reads
   are one state; keeping them all would pretend the page had been measured
   many times when it was only read many times.
2. Work in log-time, x = ln(1 + hours since the release day began), and look
   only at the recent window (the last day and a half of states): the
   direction a rate is travelling now is what the next days inherit, not the
   steep climb of launch night.
3. Two views of each trophy's future, averaged by how well each fits the
   window:
   a. a trend view: a weighted straight line of the logit's rate on x,
      anchored at the last observed rate, its slope shrunk toward its
      tier's median slope;
   b. a saturating view: the adoption curve r(t) = A * (1 - exp(-(t+2)/s)),
      fitted by a grid search over the time constant s, which is how a
      trophy's earn rate actually behaves -- it climbs toward the share of
      players who will ever earn it and then levels off.  A straight line
      in logit space would keep climbing long after the rate itself has
      levelled off, so the saturating view is what stops the forecast
      running away to 100%.  It is used only when the trophy's recent
      direction is flat or rising; a falling trophy (dilution winning over
      earning) is left to the trend view, which can fall.
4. A flat history (no refresh ever seen) gets a refresh prior instead of a
   flat line: the next refresh is expected to multiply the tracked base
   several-fold, as the kit documents it did on launch night (638 -> 3,868
   players), and new players earn each trophy at a rate proportional to how
   common it already is, so every rate is expected to fall, the common ones
   least.  A trophy the whole base already holds takes one whole trophy
   out of each new player's budget first: it stayed at 100% through a
   many-fold dilution before.
5. Spread: the window fit's residual scatter, the slope's standard error
   stretched over the extrapolation distance, the disagreement between the
   two views, and a base term growing with the square root of the horizon,
   combined in logit space.  A trophy whose rate sat at 100% through a
   many-fold dilution is forecast to stay near 100% with a long lower tail,
   not to sit exactly on 100.

Output: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# The release day begins here; log-time is measured from it.
RELEASE_EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Recency weighting: a state this many hours older than the last counts half.
HALF_LIFE_HOURS = 36.0
# The slope window: only states this many hours before the last one set the
# direction of travel.  Fewer than three states in it, and the whole history
# is used instead.
WINDOW_HOURS = 36.0
# How strongly a trophy's own slope is pulled toward its tier's median slope.
TIER_PRIOR_WEIGHT = 3.0
GLOBAL_PRIOR_WEIGHT = 1.0
# Base logit spread per sqrt(day) of horizon: the part no fit can remove.
# 0.22 was picked over 0.30 and 0.15 on synthetic final-like cases: 0.30
# leaves money on the table, 0.15 trusts the (noiseless) simulations too
# much for a real launch week.
BASE_SIGMA_PER_SQRT_DAY = 0.22
# Extra logit spread when the history never showed a refresh: the next one
# is a level shift the fit cannot see.
FLAT_HISTORY_SIGMA = 0.8
# Extra logit spread for the lumpy refresh schedule in general.
REFRESH_SIGMA = 0.15
# Time constants tried for the saturating adoption curve, in hours.
SAT_GRID = [4.0, 6.0, 9.0, 13.0, 20.0, 30.0, 45.0, 70.0, 100.0, 150.0,
            220.0, 330.0, 500.0, 750.0, 1100.0]

# Refresh prior for a flat history, calibrated on the launch-night refresh the
# kit documents: 638 -> 3,868 players tracked over a flat stretch of about
# 4.7 hours, with the new players earning ~1.03 trophies each on average.
REFRESH_FLAT_HOURS = 4.67
REFRESH_GROWTH = 6.06
NEW_PLAYER_TROPHIES = 1.03


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    # May be slightly negative: the first reads predate midnight by minutes.
    return (moment - RELEASE_EPOCH).total_seconds() / 3600.0


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def weighted_line(xs, ys, ws):
    """Weighted least squares of y on x.

    Returns (slope, intercept, resid_sd, slope_se), or None when degenerate.
    """
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    resid_sd = math.sqrt(max(var, 0.0))
    slope_se = math.sqrt(max(var, 1e-9) / sxx)
    return slope, intercept, resid_sd, slope_se


def distinct_states(case):
    """Collapse history into distinct page states, oldest first.

    A state is (hours since release, players tracked, rates tuple).  Reads
    that repeat an earlier state exactly add no information.
    """
    states = []
    seen = set()
    for read in case.get("history", []):
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if key in seen:
            continue
        seen.add(key)
        moment = parse_time(read["at"])
        states.append({
            "t": hours_since_release(moment),
            "players": read.get("players_tracked") or 0,
            "rates": list(read.get("rates") or []),
        })
    return states


def window_states(states):
    """The recent window of states that sets the direction of travel.

    States from the last WINDOW_HOURS before the final state; when that
    leaves fewer than three states, the whole history is used instead, so
    a short history still fits.
    """
    last_t = states[-1]["t"]
    window = [s for s in states if s["t"] >= last_t - WINDOW_HOURS]
    return window if len(window) >= 3 else list(states)


def smoothed_rate(rate, players):
    """A rate with half an earner of smoothing, so 0% and 100% stay finite."""
    if rate is None:
        return None
    earners = rate / 100.0 * players
    return 100.0 * (earners + 0.5) / (players + 1.0)


def saturating_fit(ts, rs, ws, t_target):
    """Fit r(t) = A * (1 - exp(-(t+2)/s)) over a grid of time constants.

    Returns (predicted rate at t_target, weighted mean squared error over
    the window), or None when there is nothing to fit.
    """
    if len(ts) < 3:
        return None
    best = None
    for s in SAT_GRID:
        us = [1.0 - math.exp(-(t + 2.0) / s) for t in ts]
        num = sum(w * u * r for w, u, r in zip(ws, us, rs))
        den = sum(w * u * u for w, u in zip(ws, us))
        if den <= 1e-12:
            continue
        a = num / den
        a = min(max(a, 0.0), 100.0)
        mse = sum(w * (r - a * u) ** 2 for w, u, r in zip(ws, us, rs)) / sum(ws)
        if best is None or mse < best[1]:
            best = (a, mse, s)
    if best is None:
        return None
    a, mse, s = best
    pred = a * (1.0 - math.exp(-(t_target + 2.0) / s))
    return min(max(pred, 0.0), 100.0), mse


def flat_refresh_median(j, last, t_target, t_first):
    """Median logit after an expected refresh, when no refresh was ever seen.

    The tracked base is expected to jump by REFRESH_GROWTH over the flat
    stretch (calibrated on the launch-night refresh the kit documents), and
    the new players are expected to earn NEW_PLAYER_TROPHIES trophies each.
    Those new trophies are spread across trophies in proportion to how
    common each already is -- except a trophy the whole base already holds:
    it stayed at 100% through a many-fold dilution before, so new players
    are expected to earn it at ~100%, and it takes one whole trophy out of
    each new player's budget before the rest is shared out.
    """
    n_last = max(last["players"], 1)
    flat_hours = max(0.0, t_target - t_first)
    growth = 1.0 + (REFRESH_GROWTH - 1.0) * (flat_hours / REFRESH_FLAT_HOURS)
    growth = min(growth, REFRESH_GROWTH * 1.2)
    n_new = n_last * growth
    dn = n_new - n_last
    rates = last["rates"]
    saturated = [k for k, r in enumerate(rates) if r is not None and r >= 99.9]
    budget = max(NEW_PLAYER_TROPHIES - len(saturated) * 1.0, 0.0)
    others_total = sum(r for k, r in enumerate(rates)
                       if r is not None and k not in saturated) or 1.0
    rate = rates[j] if j < len(rates) and rates[j] is not None else 0.0
    earners_old = rate / 100.0 * n_last
    if j in saturated:
        share = 1.0
    else:
        share = budget * rate / others_total
    earners_new = earners_old + dn * share
    new_rate = 100.0 * earners_new / n_new
    return logit_of(new_rate)


def forecast_case(case):
    trophies = case["trophies"]
    n_trophies = len(trophies)
    states = distinct_states(case)
    if not states:
        raise ValueError("case %s has no history" % case.get("id"))
    last = states[-1]
    target = parse_time(case["target"])
    t_target = hours_since_release(target)
    x_target = math.log1p(max(t_target, -0.99))
    horizon_days = max(0.0, case.get("horizon_minutes", 0) / 1440.0)

    for s in states:
        s["x"] = math.log1p(max(s["t"], -0.99))
        s["w"] = 0.5 ** ((last["t"] - s["t"]) / HALF_LIFE_HOURS)
    x_last = last["x"]
    fit_states = window_states(states)

    # --- Per-trophy fits over the recent window. ---
    trend = []       # (slope, resid_sd, slope_se, n_points, last_logit)
    sat_pred = []    # predicted rate at the target from the saturating fit
    sat_mse = []     # its window fit error, on the rate scale
    trend_mse = []   # the trend view's window fit error, on the rate scale
    for j in range(n_trophies):
        xs, ys, ws, ts, rs = [], [], [], [], []
        for s in fit_states:
            rate = s["rates"][j] if j < len(s["rates"]) else None
            if rate is None:
                continue
            xs.append(s["x"])
            ys.append(logit_of(smoothed_rate(rate, s["players"])))
            ws.append(s["w"])
            ts.append(s["t"])
            rs.append(rate)
        fit = weighted_line(xs, ys, ws) if len(xs) >= 2 else None
        last_rate = last["rates"][j] if j < len(last["rates"]) else None
        last_logit = logit_of(smoothed_rate(last_rate, last["players"])) if last_rate is not None else 0.0
        if fit is not None:
            trend.append((fit[0], fit[2], fit[3], len(xs), last_logit))
            # The trend view's window error, on the rate scale, for the
            # model average below.  xs/rs hold only the states that actually
            # listed this trophy, so nulls cannot misalign the errors.
            errs = [(r - inv_logit(last_logit + fit[0] * (x - x_last))) ** 2
                    for x, r in zip(xs, rs)]
            wsum = sum(ws)
            trend_mse.append(sum(w * e for w, e in zip(ws, errs)) / wsum if wsum > 0 else 0.0)
        else:
            trend.append((0.0, 0.25, 0.0, len(xs), last_logit))
            trend_mse.append(0.0)
        sat = saturating_fit(ts, rs, ws, t_target)
        if sat is not None:
            sat_pred.append(sat[0])
            sat_mse.append(sat[1])
        else:
            sat_pred.append(None)
            sat_mse.append(None)

    # Tier-median slopes and the global median slope, for shrinkage.
    tier_slopes = {}
    for tier in {t.get("tier") for t in trophies}:
        vals = [trend[j][0] for j, t in enumerate(trophies)
                if t.get("tier") == tier and trend[j][3] >= 3]
        if vals:
            tier_slopes[tier] = median(vals)
    global_slope = median([tr[0] for tr in trend if tr[3] >= 3]) if any(tr[3] >= 3 for tr in trend) else 0.0

    flat = len(states) <= 2

    rows = []
    for j, trophy in enumerate(trophies):
        slope, resid_sd, slope_se, n_pts, last_logit = trend[j]
        tier_prior = tier_slopes.get(trophy.get("tier"), global_slope)
        # Shrink the trophy's own slope toward its tier and the global value.
        denom = n_pts + TIER_PRIOR_WEIGHT + GLOBAL_PRIOR_WEIGHT
        shrunk_slope = (
            n_pts * slope + TIER_PRIOR_WEIGHT * tier_prior + GLOBAL_PRIOR_WEIGHT * global_slope
        ) / denom if denom > 0 else 0.0
        dx = x_target - x_last

        last_rate = last["rates"][j] if j < len(last["rates"]) else None

        # A trophy the whole tracked base already holds: it survived every
        # dilution so far, so forecast it to stay near the top, with a long
        # lower tail rather than a point mass at 100.
        if last_rate is not None and last_rate >= 99.9:
            rows.append({"id": trophy["id"], "q": [98.0, 99.5, 99.95, 99.99, 99.99]})
            continue

        if flat:
            # The refresh prior is calibrated on a short horizon (the next
            # refresh, an hour or two out).  Over a long horizon with a flat
            # history there is no trend to follow and no refresh to catch up
            # on, so fall back to persistence with a wide spread instead of
            # pretending one refresh explains days of play.
            if horizon_days <= 0.25:
                median_logit = flat_refresh_median(j, last, t_target, states[0]["t"])
            else:
                median_logit = last_logit
            sigma = math.sqrt(
                0.25 ** 2
                + (BASE_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                + FLAT_HISTORY_SIGMA ** 2
            )
        else:
            # Trend view, anchored at the last observed rate.
            trend_logit = last_logit + shrunk_slope * dx
            median_logit = trend_logit
            view_gap = 0.0
            # Saturating view, when the trophy is flat or rising: average
            # the two views by their window fit, so the view that described
            # the recent window better counts more.
            if shrunk_slope >= 0 and sat_pred[j] is not None and sat_mse[j] is not None:
                sat_logit = logit_of(sat_pred[j])
                denom_mse = sat_mse[j] + trend_mse[j]
                w_sat = trend_mse[j] / denom_mse if denom_mse > 1e-12 else 0.5
                w_sat = min(max(w_sat, 0.15), 0.85)
                median_logit = w_sat * sat_logit + (1.0 - w_sat) * trend_logit
                view_gap = abs(sat_logit - trend_logit)
            sigma = math.sqrt(
                min(max(resid_sd, 0.15), 0.8) ** 2
                + (slope_se * dx) ** 2
                + (0.5 * view_gap) ** 2
                + (BASE_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                + REFRESH_SIGMA ** 2
            )

        qs = [round4(clamp_rate(inv_logit(median_logit + z * sigma))) for z in Z]
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        raise SystemExit("no case files in %s" % INPUT_DIR)
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / ("%s.out" % path.stem)
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
