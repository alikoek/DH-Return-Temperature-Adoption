"""
Shared scenario machinery: run the logit model for every (archetype x
renovation variant), aggregate to archetypes and to the whole stock, and write
the result tables.

Aggregation (paper Eq. 5): variant shares are weighted by the number of
buildings each variant represents in the Invert building stock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load_data import MEASURE_SHORT_NAMES
from interpolate_measures import parse_building_size
from analyze_renovations import run_logit_for_variant


# -----------------------------------------------------------------------------
# Reference economics and diffusion
# -----------------------------------------------------------------------------

DISCOUNT_RATE = 0.05

# Reference case used for all per-tariff detail runs
BASE_ECONOMICS = {
    'subsidy_rate': 0.0,
    'lambda': 4,
    'energy_price': 0.10,
    'discount_rate': DISCOUNT_RATE,
}

# Cumulative share of the stock that has faced a decision occasion (Eq. 6)
DIFFUSION_RATES = {2030: 0.20, 2040: 0.60}


# -----------------------------------------------------------------------------
# Logging helper: tee stdout to a file
# -----------------------------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# -----------------------------------------------------------------------------
# Model runs
# -----------------------------------------------------------------------------

def make_baseline_measure(template_measure):
    """The do-nothing alternative: no investment, no effect."""
    n = len(template_measure['building_types'])
    return {
        'measure_name': 'Do Nothing (Baseline)',
        'short_name': MEASURE_SHORT_NAMES.get('baseline', 'Baseline'),
        'building_types': template_measure['building_types'],
        'investment_costs': np.zeros(n),
        'operational_costs': np.zeros(n),
        'lifetimes': np.ones(n) * 20.0,
        'vt_reduction': np.zeros(n),
        'rt_reduction': np.zeros(n),
        'efficiency_improvement': np.zeros(n),
    }


def aggregate_scenario_results(all_results, renovation_data, all_ids):
    """
    Aggregate per-variant results into per-archetype results.

    Weights market shares, total costs, and physical effects by the
    Invert renovation activity distribution (number_of_buildings).

    Parameters
    ----------
    all_results : dict
        (archetype, action) -> result from run_logit_for_variant()
    renovation_data : dict
        From load_renovation_data()
    all_ids : list
        All measure IDs including 'baseline'

    Returns
    -------
    dict : archetype -> {
        'weighted_shares': {mid: float},
        'weighted_costs': {mid: float},
        'weighted_rt_reduction': {mid: float},
        'weighted_vt_reduction': {mid: float},
        'weighted_efficiency': {mid: float},
        'total_buildings': float,
    }
    """
    aggregated = {}

    for archetype, class_data in renovation_data['building_classes'].items():
        total_buildings = sum(v['number_of_buildings'] for v in class_data['variants'])
        if total_buildings == 0:
            continue

        # Fraction of buildings in each renovation state
        renovation_dist = {}
        for v in class_data['variants']:
            renovation_dist[v['action']] = v['number_of_buildings'] / total_buildings

        weighted_shares = {mid: 0.0 for mid in all_ids}
        weighted_costs = {mid: 0.0 for mid in all_ids}
        weighted_rt_red = {mid: 0.0 for mid in all_ids}
        weighted_vt_red = {mid: 0.0 for mid in all_ids}
        weighted_eff = {mid: 0.0 for mid in all_ids}

        for v in class_data['variants']:
            action = v['action']
            key = (archetype, action)
            if key not in all_results:
                continue
            frac = renovation_dist[action]
            result = all_results[key]

            for mid in all_ids:
                weighted_shares[mid] += frac * result['market_shares'][mid]
                weighted_costs[mid] += frac * result['total_costs'][mid]

                # Physical effects (only for non-baseline measures)
                if mid != 'baseline' and mid in result['interpolated_effects']:
                    ie = result['interpolated_effects'][mid]
                    weighted_rt_red[mid] += frac * ie['rt_reduction']
                    weighted_vt_red[mid] += frac * ie['vt_reduction']
                    weighted_eff[mid] += frac * ie['efficiency_improvement']

        aggregated[archetype] = {
            'weighted_shares': weighted_shares,
            'weighted_costs': weighted_costs,
            'weighted_rt_reduction': weighted_rt_red,
            'weighted_vt_reduction': weighted_vt_red,
            'weighted_efficiency': weighted_eff,
            'total_buildings': total_buildings,
        }

    return aggregated


def compute_overall_shares(aggregated, all_ids):
    """Stock-weighted shares across all archetypes."""
    total_all = sum(a['total_buildings'] for a in aggregated.values())
    overall = {mid: 0.0 for mid in all_ids}
    if total_all == 0:
        return overall
    for agg in aggregated.values():
        w = agg['total_buildings'] / total_all
        for mid in all_ids:
            overall[mid] += agg['weighted_shares'][mid] * w
    return overall


def run_one_scenario(all_measures, interp_table, renovation_data, assumptions):
    """Run logit for all (building x variant); return (all_results, aggregated, overall)."""
    measure_ids = [mid for mid in all_measures.keys() if mid != 'baseline']
    all_ids = measure_ids + ['baseline']

    all_results = {}
    for archetype, class_data in renovation_data['building_classes'].items():
        building_size = parse_building_size(archetype)
        for variant in class_data['variants']:
            result = run_logit_for_variant(
                all_measures, interp_table, archetype, variant,
                assumptions, building_size
            )
            result['archetype'] = archetype
            result['action'] = variant['action']
            result['hwb_norm'] = variant['hwb_norm']
            result['fed_sh_per_bssh'] = variant['fed_sh_per_bssh']
            result['number_of_buildings'] = variant['number_of_buildings']
            all_results[(archetype, variant['action'])] = result

    aggregated = aggregate_scenario_results(all_results, renovation_data, all_ids)
    overall = compute_overall_shares(aggregated, all_ids)
    return all_results, aggregated, overall


# -----------------------------------------------------------------------------
# Result tables
# -----------------------------------------------------------------------------

def write_per_variant_csv(out_dir, year, all_results, all_ids):
    rows = []
    short = MEASURE_SHORT_NAMES
    for (archetype, action), r in sorted(all_results.items()):
        row = {
            'building_class': archetype,
            'renovation_action': action,
            'hwb_norm': r['hwb_norm'],
            'fed_sh_per_bssh': r['fed_sh_per_bssh'],
            'number_of_buildings': r['number_of_buildings'],
        }
        for mid in all_ids:
            name = short.get(mid, mid)
            row[f'share_{name}'] = r['market_shares'][mid]
            row[f'cost_{name}'] = r['total_costs'][mid]
        rows.append(row)
    df = pd.DataFrame(rows)
    fname = out_dir / f'results_renovation_analysis_{year}.csv'
    df.to_csv(fname, index=False)
    print(f"    [OK] {fname}")


def write_aggregated_csv(out_dir, year, aggregated, overall_shares, all_ids):
    short = MEASURE_SHORT_NAMES
    total_all = sum(a['total_buildings'] for a in aggregated.values())
    rows = []
    for archetype, agg in sorted(aggregated.items()):
        row = {'building_class': archetype, 'total_buildings_invert': agg['total_buildings']}
        for mid in all_ids:
            row[f'share_{short.get(mid, mid)}'] = agg['weighted_shares'][mid]
        rows.append(row)
    overall_row = {'building_class': 'OVERALL (stock-weighted)', 'total_buildings_invert': total_all}
    for mid in all_ids:
        overall_row[f'share_{short.get(mid, mid)}'] = overall_shares[mid]
    rows.append(overall_row)
    df = pd.DataFrame(rows)
    fname = out_dir / f'results_aggregated_{year}.csv'
    df.to_csv(fname, index=False)
    print(f"    [OK] {fname}")
