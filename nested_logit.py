"""
DeRiskDH - Nested Logit (robustness check for the MNL adoption model)
=====================================================================

WHY THIS MODULE EXISTS
----------------------
The main model (logit_model.py) is a multinomial logit (MNL). MNL carries the
Independence of Irrelevant Alternatives (IIA) property: the share ratio of any
two alternatives is unaffected by the other alternatives on the menu. Our
choice set violates the spirit of IIA because the combination measures
literally contain their component measures:

    M7 (2.8)  = pump downsizing + flow limiter          (contains M1)
    M8 (2.9)  = pump + flow limiter + hydr. balancing   (contains M1, M3)
    M9 (2.10) = hydr. balancing + heat exchanger        (contains M2, M3)

A nested logit (NL) relaxes IIA by grouping similar alternatives into nests.
Within a nest, alternatives are correlated substitutes; the nesting parameter
theta in (0, 1] controls how strongly:

    theta = 1   -> no within-nest correlation, NL collapses EXACTLY to MNL
    theta -> 0  -> alternatives in the nest are perfect duplicates
                   (red-bus/blue-bus limit)

Because no observed adoption data exist to estimate theta (the same reason
lambda is scenario-based), NL is used here only as a ROBUSTNESS CHECK with
assumed theta values, not as the main model.

FORMULATION (utility-scale convention)
--------------------------------------
Using the same cost-relative utilities as the MNL:

    V_i = -lambda * c_i / c_bar

with c_bar the converged MNL weighted-average cost (the normalization is kept
fixed at its MNL value so that the substitution structure - not the cost
scaling device - is what changes between MNL and NL; see iterate_nested_logit
for the fixed-point variant used as a consistency check).

For nest m with members M_m:

    P(i | m) = exp(V_i / theta) / sum_{j in M_m} exp(V_j / theta)
    IV_m     = ln sum_{j in M_m} exp(V_j / theta)        (inclusive value)
    P(m)     = exp(theta * IV_m) / sum_n exp(theta * IV_n)
    P(i)     = P(m(i)) * P(i | m)

Singleton nests (e.g. the Do-Nothing baseline) reduce to P(i|m) = 1 and
theta * IV_m = V_i, so their probability is theta-independent, as it must be.
"""
from __future__ import annotations

import numpy as np

from logit_model import iterate_market_shares

#: Partition A - by menu structure: do-nothing / single measures / combinations
NEST_PARTITION_A = {
    'do_nothing': ['baseline'],
    'singles': ['1.3', '1.5', '2.1', '2.3', '2.5', '2.7'],
    'combos': ['2.8', '2.9', '2.10'],
}

#: Partition B - by dominant component family.
#: NOTE: M8 (2.9 = pump + flow limiter + balancing) spans two families; it is
#: assigned to 'flow_pump' here because its largest single cost component is
#: the pump replacement. This assignment needs co-author sign-off; the fact
#: that no clean partition exists is exactly why cross-nested logit is the
#: "correct" structure (future work, requires adoption data to estimate).
NEST_PARTITION_B = {
    'flow_pump': ['1.3', '2.3', '2.8'],
    'balancing': ['2.1', '2.9'],
    'heat_exchanger': ['1.5', '2.10'],
    'passive_behavioral': ['2.5', '2.7'],
    'do_nothing': ['baseline'],
}

PARTITIONS = {'A': NEST_PARTITION_A, 'B': NEST_PARTITION_B}


def calculate_nested_logit_shares(costs, lambda_param, ids, nests, theta,
                                  weighted_avg_cost):
    """
    Nested-logit choice probabilities on cost-relative utilities.

    Parameters
    ----------
    costs : array-like
        Total annual heating cost per alternative (EUR/year), aligned with ids.
    lambda_param : float
        Cost-sensitivity parameter (same role as in the MNL).
    ids : list of str
        Alternative IDs aligned with costs (e.g. ['1.3', ..., 'baseline']).
    nests : dict
        Nest name -> list of member IDs. Members not present in `ids` are
        ignored (supports restricted measure sets). Every ID in `ids` must be
        covered by exactly one nest.
    theta : float
        Nesting parameter in (0, 1], applied to all nests. theta=1 -> MNL.
    weighted_avg_cost : float
        Normalization c_bar (take from the converged MNL run).

    Returns
    -------
    np.ndarray of shares aligned with ids (sums to 1.0).
    """
    costs = np.asarray(costs, dtype=float)
    if not 0.0 < theta <= 1.0:
        raise ValueError(f"theta must be in (0, 1], got {theta}")

    # Check the partition covers all alternatives exactly once
    id_to_pos = {mid: i for i, mid in enumerate(ids)}
    covered = [mid for members in nests.values() for mid in members
               if mid in id_to_pos]
    if sorted(covered) != sorted(ids):
        missing = set(ids) - set(covered)
        dupes = {x for x in covered if covered.count(x) > 1}
        raise ValueError(f"Nest partition invalid: missing={missing}, duplicated={dupes}")

    V = -lambda_param * costs / weighted_avg_cost

    # Numerically stable two-level logit via per-nest logsumexp
    nest_names = []
    nest_logsum = []          # theta * IV_m per nest
    nest_members = []         # positions of members
    nest_cond = []            # conditional within-nest probabilities
    for name, members in nests.items():
        pos = [id_to_pos[m] for m in members if m in id_to_pos]
        if not pos:
            continue
        v = V[pos] / theta
        vmax = np.max(v)
        expv = np.exp(v - vmax)
        iv = vmax + np.log(np.sum(expv))          # IV_m = logsumexp(V/theta)
        nest_names.append(name)
        nest_logsum.append(theta * iv)
        nest_members.append(pos)
        nest_cond.append(expv / np.sum(expv))

    nest_logsum = np.array(nest_logsum)
    lmax = np.max(nest_logsum)
    nest_prob = np.exp(nest_logsum - lmax)
    nest_prob = nest_prob / np.sum(nest_prob)

    shares = np.zeros(len(costs))
    for p_m, pos, cond in zip(nest_prob, nest_members, nest_cond):
        for j, p in zip(pos, cond):
            shares[j] = p_m * p
    return shares


def iterate_nested_logit(costs, lambda_param, ids, nests, theta,
                         max_iterations=50, tolerance=1e-6):
    """
    Fixed-point variant: re-derive c_bar from the NL shares themselves and
    iterate to convergence (analogous to logit_model.iterate_market_shares).

    Used only as a consistency check on the simpler 'MNL-anchored c_bar'
    approach; the two should agree closely because c_bar only rescales the
    common cost normalization.
    """
    costs = np.asarray(costs, dtype=float)
    shares = np.ones(len(costs)) / len(costs)
    for iteration in range(max_iterations):
        c_bar = float(np.sum(shares * costs))
        new_shares = calculate_nested_logit_shares(
            costs, lambda_param, ids, nests, theta, c_bar)
        if np.max(np.abs(new_shares - shares)) < tolerance:
            return {'market_shares': new_shares, 'weighted_avg_cost': c_bar,
                    'iterations': iteration + 1, 'converged': True}
        shares = new_shares
    return {'market_shares': shares, 'weighted_avg_cost': c_bar,
            'iterations': max_iterations, 'converged': False}
