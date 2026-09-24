"""net_mag — label-free, pre-deployment diagnostic for Mondrian partition signals.

Implements the "net tightening magnitude" from the manuscript (SI §S5.2, Eq. (5)):

    net_mag(G; s) = E_{i in G}[ max(0, (q_{k(i)} - q_bar) / q_bar) ]
                  - E_{i in G}[ max(0, (q_bar - q_{k(i)}) / q_bar) ]

where ``q_bar`` is the global split-conformal quantile of the calibration
nonconformity scores and ``q_{k(i)}`` is the quantile of the Mondrian bin that
signal ``s`` assigns to molecule ``i``.

Interpretation
--------------
- ``net_mag > 0``  the subgroup is **net-widened**  -> coverage is expected to RISE;
- ``net_mag < 0``  the subgroup is **net-tightened** -> coverage is expected to FALL
  (harm warning). On all 16 (signal x representation) units of the study the sign
  of ``net_mag`` matched the sign of the realised coverage change; |net_mag|
  below ~0.5 (in percent-of-q_bar units, i.e. the value x 100) indicated no
  practically meaningful effect.

Both ``q_bar`` and ``q_k`` depend only on the **calibration** set and its labels;
assigning a target molecule to a bin needs only the signal value of that molecule.
Therefore ``net_mag`` **never touches test labels** and is computable before
deployment, on any calibration set / training fold you already have.

This module is intentionally NumPy-only (no RDKit, no scikit-learn) so it can be
dropped into any pipeline. A runnable walkthrough lives in
``notebooks/net_mag_demo.ipynb``.

Reference
---------
Manuscript: "How the coverage guarantee of conformal prediction fails on
activity cliffs" — coverage-transfer identity (Proposition 1) and the label-free
diagnostic (SI §S5.2).
"""

from __future__ import annotations

import numpy as np

__all__ = ["conformal_quantile", "mondrian_quantiles", "net_mag"]


def conformal_quantile(scores, alpha: float = 0.10) -> float:
    """Split-conformal quantile with finite-sample correction.

    Returns the smallest empirical quantile of ``scores`` such that
    P(score <= q) >= 1 - alpha under exchangeability, i.e. the
    ceil((n + 1)(1 - alpha)) - th order statistic (clamped to the maximum).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        raise ValueError("scores must be non-empty")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def mondrian_quantiles(
    cal_scores,
    cal_signal,
    n_bins: int = 3,
    alpha: float = 0.10,
):
    """Per-bin conformal quantiles for a Mondrian partition of the calibration set.

    Bins are equal-count bins of the *signal* values on the calibration set.

    Returns
    -------
    edges : (n_bins - 1,) float array — interior bin edges on the signal axis
    q_bins : (n_bins,) float array — conformal quantile per bin
    bin_id : (n_cal,) int array — bin index of each calibration molecule
    q_bar : float — global conformal quantile
    """
    sc = np.asarray(cal_scores, dtype=np.float64).ravel()
    sg = np.asarray(cal_signal, dtype=np.float64).ravel()
    if sc.shape != sg.shape:
        raise ValueError("cal_scores and cal_signal must have the same shape")

    q_bar = conformal_quantile(sc, alpha)
    edges = np.quantile(sg, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    bin_id = np.searchsorted(edges, sg, side="right")

    q_bins = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        m = bin_id == b
        if not m.any():
            q_bins[b] = q_bar  # empty bin: fall back to the global quantile
        else:
            q_bins[b] = conformal_quantile(sc[m], alpha)
    return edges, q_bins, bin_id, q_bar


def net_mag(
    target_signal,
    cal_scores,
    cal_signal,
    group=None,
    n_bins: int = 3,
    alpha: float = 0.10,
    return_details: bool = False,
):
    """Label-free pre-deployment diagnostic for a Mondrian partition signal.

    Parameters
    ----------
    target_signal : array of shape (n_target,)
        Signal value for every molecule of the (candidate) test/deployment set.
        Must be computable without test labels — e.g. local label dispersion in
        the training neighbourhood, model variance, or training-neighbour count.
    cal_scores : array of shape (n_cal,)
        Nonconformity scores |y - yhat| on the calibration set.
    cal_signal : array of shape (n_cal,)
        Same signal evaluated on the calibration set.
    group : None, boolean mask or integer indices of shape over n_target
        The subgroup G whose coverage change you care about (e.g. activity
        cliffs, a priority scaffold, a toxic substructure). ``None`` = all
        target molecules.
    n_bins : int
        Number of Mondrian bins (the manuscript recommends 3 or 5).
    alpha : float
        Miscoverage level (paper uses 0.10).
    return_details : bool
        If True, also return a dict with ``q_bar``, ``edges``, ``q_bins``,
        ``bin_id`` (per-target bin assignment) and ``contrib`` (per-target
        net contribution to the mean).

    Returns
    -------
    nm : float
        net_mag of the subgroup, in units of q_bar (multiply by 100 for
        percent-of-q_bar). Negative = net-tightened = expected coverage LOSS
        for the subgroup under this signal; positive = net-widened = expected
        benefit; |nm| near zero = no practically meaningful effect.
    details : dict (only if return_details=True)
    """
    tg = np.asarray(target_signal, dtype=np.float64).ravel()
    edges, q_bins, _, q_bar = mondrian_quantiles(cal_scores, cal_signal, n_bins, alpha)

    # assign each target molecule to its Mondrian bin (signal value only)
    bin_t = np.searchsorted(edges, tg, side="right")
    q_t = q_bins[bin_t]

    # per-molecule net movement of its assigned quantile, relative to q_bar
    rel = (q_t - q_bar) / q_bar
    contrib = np.maximum(rel, 0.0) - np.maximum(-rel, 0.0)

    if group is None:
        sel = np.ones(tg.size, dtype=bool)
    else:
        sel = np.asarray(group)
        if sel.dtype == bool:
            if sel.shape != tg.shape:
                raise ValueError("boolean `group` mask must match target_signal")
        else:
            sel = np.zeros(tg.size, dtype=bool)
            sel[np.asarray(group, dtype=np.int64)] = True
        if not sel.any():
            raise ValueError("`group` selects no molecules")

    nm = float(contrib[sel].mean())

    if return_details:
        details = {
            "q_bar": q_bar,
            "edges": edges,
            "q_bins": q_bins,
            "bin_id": bin_t,
            "contrib": contrib,
            "group_mask": sel,
            "n_group": int(sel.sum()),
            "alpha": alpha,
            "n_bins": n_bins,
        }
        return nm, details
    return nm
