"""
Step 2 of 4: reference case, cost decomposition, sensitivities and the
nested-logit robustness check.

Parts
-----
1. Reference case (all nine measures + do-nothing; no tariff; lambda = 4;
   0.10 EUR/kWh; no subsidy; r = 5 %) for 2030 and 2040, checked against the
   no-tariff run of run_scenarios.py (the two scripts must agree exactly).
2. Stock-weighted cost decomposition (energy / annualized investment / O&M)
   per alternative.
3. Sensitivity to the flat efficiency assumption of the user-behavior
   measure (M6): 5.0 / 9.6 / 15.0 %.
4. Nested-logit robustness: choice-structure partition (A) and
   component-family partition (B) for theta in {0.3, 0.5, 0.7} plus the
   theta = 1 control, including a fixed-point consistency check.
5. Lambda / energy-price / subsidy sensitivity and the diffusion-adjusted
   stock adoption (d = 20 % by 2030, 60 % by 2040).

Requires outputs/scenarios/ from run_scenarios.py.
Outputs land in outputs/paper_analysis/; stdout is also written to run.log.

Run:  python run_paper_analysis.py
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import load_all_measures, MEASURE_SHORT_NAMES
from load_invert_data import load_renovation_data
from interpolate_measures import build_interpolation_table
from analyze_renovations import build_base_hwb_mapping
from scenario import (
    BASE_ECONOMICS,
    DIFFUSION_RATES,
    Tee,
    aggregate_scenario_results,
    make_baseline_measure,
    run_one_scenario,
    write_aggregated_csv,
    write_per_variant_csv,
)
from logit_model import iterate_market_shares
from nested_logit import (
    PARTITIONS,
    calculate_nested_logit_shares,
    iterate_nested_logit,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
YEARS = [2030, 2040]
SCENARIO_NONE_DIR = ROOT / 'outputs' / 'scenarios' / 'all_measures' / 'per_tariff' / 'none'
OUT_ROOT = ROOT / 'outputs' / 'paper_analysis'

M6_EFFICIENCIES = [5.0, 9.6, 15.0]    # flat % override for measure 2.7
NL_THETAS = [0.3, 0.5, 0.7, 1.0]      # 1.0 = MNL control row
LAMBDA_GRID = [2, 4, 8]               # 4 = reference
PRICE_GRID = [0.08, 0.10, 0.12]       # 0.10 = reference
SUBSIDY_GRID = [0.0, 0.30]            # 0.0 = reference
ASSERT_TOL = 1e-9


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def setup_measures():
    """Load the full 9-measure set and append the do-nothing alternative."""
    all_measures = load_all_measures()
    first = next(iter(all_measures.values()))
    all_measures['baseline'] = make_baseline_measure(first)
    all_ids = [m for m in all_measures.keys() if m != 'baseline'] + ['baseline']
    return all_measures, all_ids, first['building_types']


def run_base_for_year(all_measures, renovation_data):
    interp = build_interpolation_table(all_measures, build_base_hwb_mapping(renovation_data))
    return run_one_scenario(all_measures, interp, renovation_data, dict(BASE_ECONOMICS))


def assert_matches_scenario_run(overall, all_ids, year):
    """Compare stock-weighted reference shares with the run_scenarios.py output."""
    ref_file = SCENARIO_NONE_DIR / f'results_aggregated_{year}.csv'
    ref = pd.read_csv(ref_file)
    row = ref[ref['building_class'] == 'OVERALL (stock-weighted)'].iloc[0]
    worst = 0.0
    for mid in all_ids:
        col = f"share_{MEASURE_SHORT_NAMES.get(mid, mid)}"
        diff = abs(float(row[col]) - overall[mid])
        worst = max(worst, diff)
    if worst > ASSERT_TOL:
        raise AssertionError(
            f"Reference case {year} differs from {ref_file} by up to {worst:.2e} "
            f"(tolerance {ASSERT_TOL:.0e}). Re-run run_scenarios.py first.")
    print(f"  [ASSERT OK] {year}: matches {ref_file} (max diff {worst:.2e})")


def cost_decomposition(all_results, all_ids, out_csv):
    """Stock-weighted (by number_of_buildings) cost components per alternative."""
    weights, comp = [], {mid: {'energy': [], 'inv': [], 'om': [], 'total': []} for mid in all_ids}
    for (_, _), r in sorted(all_results.items()):
        weights.append(r['number_of_buildings'])
        for mid in all_ids:
            cd = r['cost_details'][mid]
            om = cd['total_annual_cost'] - cd['energy_cost'] - cd['annualized_investment']
            comp[mid]['energy'].append(cd['energy_cost'])
            comp[mid]['inv'].append(cd['annualized_investment'])
            comp[mid]['om'].append(om)
            comp[mid]['total'].append(cd['total_annual_cost'])
    w = np.array(weights)
    rows = []
    for mid in all_ids:
        rows.append({
            'measure_id': mid,
            'measure': MEASURE_SHORT_NAMES.get(mid, mid),
            'energy_cost': np.average(comp[mid]['energy'], weights=w),
            'annualized_investment': np.average(comp[mid]['inv'], weights=w),
            'om_cost': np.average(comp[mid]['om'], weights=w),
            'total_cost': np.average(comp[mid]['total'], weights=w),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [OK] {out_csv}")


def overall_from_modified_shares(all_results, renovation_data, all_ids, share_fn):
    """Clone per-variant results, replace market_shares via share_fn, re-aggregate."""
    modified = {}
    for key, r in all_results.items():
        costs = np.array([r['total_costs'][mid] for mid in all_ids])
        new_shares = share_fn(costs)
        clone = dict(r)
        clone['market_shares'] = {mid: new_shares[i] for i, mid in enumerate(all_ids)}
        modified[key] = clone
    agg = aggregate_scenario_results(modified, renovation_data, all_ids)
    total = sum(a['total_buildings'] for a in agg.values())
    overall = {mid: 0.0 for mid in all_ids}
    for a in agg.values():
        w = a['total_buildings'] / total
        for mid in all_ids:
            overall[mid] += a['weighted_shares'][mid] * w
    return overall


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    for sub in ('base_case', 'm6_sensitivity', 'nested_logit', 'sensitivity'):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    print("#" * 80)
    print("#  PAPER ANALYSIS RUNS")
    print(f"#  Output root: {OUT_ROOT}")
    print("#" * 80)

    all_measures, all_ids, archetype_types = setup_measures()
    lambda_base = BASE_ECONOMICS['lambda']

    renovation = {y: load_renovation_data(year=y, archetype_types=archetype_types)
                  for y in YEARS}

    # ------------------------------------------------------------------
    # Part 1: reference case + consistency check
    # ------------------------------------------------------------------
    print("\n=== PART 1: reference case (2030, 2040) + consistency check ===")
    base = {}
    for year in YEARS:
        all_results, agg, overall = run_base_for_year(all_measures, renovation[year])
        base[year] = {'all_results': all_results, 'aggregated': agg, 'overall': overall}
        assert_matches_scenario_run(overall, all_ids, year)
        write_per_variant_csv(OUT_ROOT / 'base_case', year, all_results, all_ids)
        write_aggregated_csv(OUT_ROOT / 'base_case', year, agg, overall, all_ids)

    # ------------------------------------------------------------------
    # Part 2: cost decomposition
    # ------------------------------------------------------------------
    print("\n=== PART 2: stock-weighted cost decomposition ===")
    for year in YEARS:
        cost_decomposition(base[year]['all_results'], all_ids,
                           OUT_ROOT / 'base_case' / f'cost_decomposition_{year}.csv')

    # ------------------------------------------------------------------
    # Part 3: M6 efficiency sensitivity
    # ------------------------------------------------------------------
    print("\n=== PART 3: M6 (2.7) flat-efficiency sensitivity ===")
    m6_rows = []
    for eff in M6_EFFICIENCIES:
        measures_mod = copy.deepcopy(all_measures)
        measures_mod['2.7']['efficiency_improvement'][:] = eff
        for year in YEARS:
            interp = build_interpolation_table(
                measures_mod, build_base_hwb_mapping(renovation[year]))
            _, agg, overall = run_one_scenario(
                measures_mod, interp, renovation[year], dict(BASE_ECONOMICS))
            out_dir = OUT_ROOT / 'm6_sensitivity' / f'm6_eff_{eff:g}'
            out_dir.mkdir(exist_ok=True)
            write_aggregated_csv(out_dir, year, agg, overall, all_ids)
            for mid in all_ids:
                m6_rows.append({'m6_efficiency_pct': eff, 'year': year,
                                'measure_id': mid,
                                'measure': MEASURE_SHORT_NAMES.get(mid, mid),
                                'stock_weighted_share': overall[mid]})
            print(f"  m6_eff={eff:g}%  year={year}:  M6 share = {overall['2.7']:.4f},"
                  f"  Do-Nothing = {overall['baseline']:.4f}")
    m6_csv = OUT_ROOT / 'm6_sensitivity' / 'm6_sensitivity_summary.csv'
    pd.DataFrame(m6_rows).to_csv(m6_csv, index=False)
    print(f"  [OK] {m6_csv}")

    # ------------------------------------------------------------------
    # Part 4: nested-logit robustness
    # ------------------------------------------------------------------
    print("\n=== PART 4: nested-logit robustness (partitions A/B, theta grid) ===")
    nl_rows = []
    for year in YEARS:
        all_results = base[year]['all_results']
        mnl_overall = base[year]['overall']
        mnl_rank = {mid: rank for rank, mid in enumerate(
            sorted(all_ids, key=lambda m: -mnl_overall[m]), start=1)}
        for pname, partition in PARTITIONS.items():
            for theta in NL_THETAS:
                def nl_fn(costs, _p=partition, _t=theta):
                    mnl = iterate_market_shares(costs, lambda_param=lambda_base)
                    return calculate_nested_logit_shares(
                        costs, lambda_base, all_ids, _p, _t,
                        weighted_avg_cost=mnl['weighted_avg_cost'])
                overall = overall_from_modified_shares(
                    all_results, renovation[year], all_ids, nl_fn)
                ranks = {mid: rank for rank, mid in enumerate(
                    sorted(all_ids, key=lambda m: -overall[m]), start=1)}
                for mid in all_ids:
                    nl_rows.append({
                        'year': year, 'partition': pname, 'theta': theta,
                        'measure_id': mid,
                        'measure': MEASURE_SHORT_NAMES.get(mid, mid),
                        'stock_weighted_share': overall[mid],
                        'rank': ranks[mid],
                        'mnl_share': mnl_overall[mid],
                        'mnl_rank': mnl_rank[mid],
                        'delta_pp': (overall[mid] - mnl_overall[mid]) * 100,
                    })
                top3 = sorted(all_ids, key=lambda m: -overall[m])[:3]
                print(f"  year={year} partition={pname} theta={theta}: top3 = "
                      f"{[MEASURE_SHORT_NAMES.get(m, m) for m in top3]}")
    nl_csv = OUT_ROOT / 'nested_logit' / 'nl_robustness.csv'
    pd.DataFrame(nl_rows).to_csv(nl_csv, index=False)
    print(f"  [OK] {nl_csv}")

    # Fixed-point consistency check (partition A, theta=0.5, 2030):
    # MNL-anchored c_bar (single NL application) vs full NL fixed point.
    print("\n  Fixed-point consistency check (partition A, theta=0.5, 2030):")
    partition = PARTITIONS['A']
    worst = 0.0
    for (_, _), r in base[2030]['all_results'].items():
        costs = np.array([r['total_costs'][mid] for mid in all_ids])
        mnl = iterate_market_shares(costs, lambda_param=lambda_base)
        single = calculate_nested_logit_shares(
            costs, lambda_base, all_ids, partition, 0.5,
            weighted_avg_cost=mnl['weighted_avg_cost'])
        fixed = iterate_nested_logit(costs, lambda_base, all_ids, partition, 0.5)
        worst = max(worst, float(np.max(np.abs(single - fixed['market_shares']))))
    print(f"  [CHECK] max |single-application - fixed-point| share difference "
          f"across all 45 variants: {worst:.2e} ({worst*100:.4f} pp)")

    # ------------------------------------------------------------------
    # Part 5: lambda / price / subsidy sensitivity + diffusion
    # ------------------------------------------------------------------
    print("\n=== PART 5: lambda / energy-price / subsidy sensitivity + diffusion ===")
    sens_rows = []

    def record(param, value, year, overall):
        for mid in all_ids:
            sens_rows.append({'param': param, 'value': value, 'year': year,
                              'measure_id': mid,
                              'measure': MEASURE_SHORT_NAMES.get(mid, mid),
                              'stock_weighted_share': overall[mid]})

    for year in YEARS:
        interp = build_interpolation_table(
            all_measures, build_base_hwb_mapping(renovation[year]))
        for lam in LAMBDA_GRID:
            if lam == BASE_ECONOMICS['lambda']:
                record('lambda', lam, year, base[year]['overall'])
                continue
            a = dict(BASE_ECONOMICS); a['lambda'] = lam
            _, _, overall = run_one_scenario(all_measures, interp, renovation[year], a)
            record('lambda', lam, year, overall)
        for price in PRICE_GRID:
            if price == BASE_ECONOMICS['energy_price']:
                record('energy_price', price, year, base[year]['overall'])
                continue
            a = dict(BASE_ECONOMICS); a['energy_price'] = price
            _, _, overall = run_one_scenario(all_measures, interp, renovation[year], a)
            record('energy_price', price, year, overall)
        for subsidy in SUBSIDY_GRID:
            if subsidy == BASE_ECONOMICS['subsidy_rate']:
                record('subsidy', subsidy, year, base[year]['overall'])
                continue
            a = dict(BASE_ECONOMICS); a['subsidy_rate'] = subsidy
            _, _, overall = run_one_scenario(all_measures, interp, renovation[year], a)
            record('subsidy', subsidy, year, overall)
    sens_csv = OUT_ROOT / 'sensitivity' / 'lambda_price_sensitivity.csv'
    pd.DataFrame(sens_rows).to_csv(sens_csv, index=False)
    print(f"  [OK] {sens_csv}")

    diff_rows = []
    for year in YEARS:
        d = DIFFUSION_RATES[year]
        for mid in all_ids:
            s = base[year]['overall'][mid]
            adjusted = s * d + (1 - d) if mid == 'baseline' else s * d
            diff_rows.append({'year': year, 'diffusion_rate': d, 'measure_id': mid,
                              'measure': MEASURE_SHORT_NAMES.get(mid, mid),
                              'choice_share': s, 'stock_adoption': adjusted})
    diff_csv = OUT_ROOT / 'base_case' / 'overall_shares_with_diffusion.csv'
    pd.DataFrame(diff_rows).to_csv(diff_csv, index=False)
    print(f"  [OK] {diff_csv}")

    print(f"\nDone in {time.time() - t0:.1f} s. All outputs under {OUT_ROOT}")


if __name__ == '__main__':
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = open(OUT_ROOT / 'run.log', 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
