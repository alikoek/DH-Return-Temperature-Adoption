"""
Step 1 of 4: motivational-tariff scenarios.

For each choice set and stock year (2030, 2040), runs the reference economics
(no subsidy, lambda = 4, 0.10 EUR/kWh, r = 5 %) under

  - no tariff,
  - four stepped bonus/malus rates b = 5, 10, 15, 20 % (35/39 degC thresholds),
  - five linear slopes a = 0.2 ... 1.0 EUR/MWh per K (reference 37 degC).

Choice sets:
  all_measures        all nine measures + do-nothing (main analysis)
  restricted          without pump downsizing, buffer storage and user
                      behavior (Appendix B)

Outputs (outputs/scenarios/<choice set>/):
  per_tariff/<tariff>/results_renovation_analysis_<year>.csv  per variant
  per_tariff/<tariff>/results_aggregated_<year>.csv           per archetype + stock
  sweep_summary/sweep_linear.csv, sweep_stepped.csv           stock shares vs intensity
  expected_rt_reduction.csv                                   E[dT_RT] per tariff

Run:  python run_scenarios.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import load_all_measures, load_building_baseline_rt, MEASURE_SHORT_NAMES
from load_invert_data import load_renovation_data
from interpolate_measures import build_interpolation_table
from analyze_renovations import build_base_hwb_mapping
from scenario import (
    BASE_ECONOMICS,
    Tee,
    make_baseline_measure,
    run_one_scenario,
    write_aggregated_csv,
    write_per_variant_csv,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LINEAR_SLOPES = [0.0002, 0.0004, 0.0006, 0.0008, 0.0010]  # EUR/kWh/K (= 0.2..1.0 EUR/MWh/K)
STEPPED_RATES = [0.05, 0.10, 0.15, 0.20]                   # bonus/malus (symmetric)
REFERENCE_RT = 37.0

MEASURE_SETS = [
    {'name': 'all_measures',       'excluded': []},
    {'name': 'restricted',         'excluded': ['2.3', '2.5', '2.7']},  # drop M4, M5, M6
]
YEARS = [2030, 2040]

OUTPUT_ROOT = Path(__file__).resolve().parent / 'outputs' / 'scenarios'


def build_tariff_configs():
    """Tariff configurations: none + 4 stepped + 5 linear."""
    configs = [{'label': 'none', 'kind': 'none', 'level': None, 'params': None}]
    for rate in STEPPED_RATES:
        configs.append({
            'label': f'stepped {int(rate*100)}%',
            'kind': 'stepped',
            'level': rate,
            'params': {
                'approach': 'stepped',
                'bonus_threshold': 35, 'malus_threshold': 39,
                'bonus_rate': rate, 'malus_rate': rate,
            },
        })
    for slope in LINEAR_SLOPES:
        configs.append({
            'label': f'linear {slope:.4f} EUR/(kWh*K)',
            'kind': 'linear',
            'level': slope,
            'params': {
                'approach': 'linear',
                'bonus_malus_factor': slope,
                'reference_rt': REFERENCE_RT,
            },
        })
    return configs


def tariff_folder(tariff):
    """Short folder name: none, stepped_<b in %>pct, linear_<a in EUR/MWh per K>."""
    if tariff['kind'] == 'linear':
        return f"linear_{tariff['level'] * 1000:g}"
    if tariff['kind'] == 'stepped':
        return f"stepped_{tariff['level'] * 100:g}pct"
    return 'none'


def expected_rt_reduction(aggregated, all_ids):
    """Expected return-temperature reduction per building facing a decision
    (K): sum over archetypes b and alternatives m of w_b * s_bm * dT_RT,bm,
    with building-number weights w_b and no diffusion adjustment."""
    num, den = 0.0, 0.0
    for data in aggregated.values():
        w = data['total_buildings']
        e_rt = sum(data['weighted_shares'][mid] * data['weighted_rt_reduction'][mid]
                   for mid in all_ids)
        num += w * e_rt
        den += w
    return num / den


def run_for_measure_set(measure_set_root, excluded_ids, baseline_rts_cache=None):
    """Run every tariff configuration for one choice set."""
    tariff_configs = build_tariff_configs()

    print("=" * 80)
    print(f"CHOICE SET: {measure_set_root.name}")
    print(f"Excluded measures: {excluded_ids if excluded_ids else '(none - all 9 measures kept)'}")
    print(f"Tariff configs: {len(tariff_configs)} (1 none + {len(STEPPED_RATES)} stepped "
          f"+ {len(LINEAR_SLOPES)} linear)")
    print("=" * 80)

    t_start = time.time()

    all_measures = load_all_measures()
    for mid in excluded_ids:
        if mid in all_measures:
            print(f"  Excluding measure {mid}")
            all_measures.pop(mid)

    first_measure = next(iter(all_measures.values()))
    archetype_types = first_measure['building_types']
    all_measures['baseline'] = make_baseline_measure(first_measure)
    baseline_rts = (baseline_rts_cache if baseline_rts_cache is not None
                    else load_building_baseline_rt())

    measure_ids = [mid for mid in all_measures.keys() if mid != 'baseline']
    all_ids = measure_ids + ['baseline']
    print(f"\nKept measures: {[MEASURE_SHORT_NAMES.get(m, m) for m in all_ids]}")

    sweep_linear_rows = []
    sweep_stepped_rows = []
    ert_rows = []

    for year in YEARS:
        print(f"\n{'#' * 80}\n#  YEAR {year}\n{'#' * 80}")
        renovation_data = load_renovation_data(year=year, archetype_types=archetype_types)
        interp_table = build_interpolation_table(
            all_measures, build_base_hwb_mapping(renovation_data)
        )

        for tariff in tariff_configs:
            tariff_label = tariff['label']
            tariff_dir = measure_set_root / 'per_tariff' / tariff_folder(tariff)
            tariff_dir.mkdir(parents=True, exist_ok=True)

            assumptions = dict(BASE_ECONOMICS)
            if tariff['params'] is not None:
                params = dict(tariff['params'])
                params['building_baseline_rts'] = baseline_rts
                assumptions['motivational_tariff_params'] = params

            print(f"\n--- Year {year}, tariff '{tariff_label}' ---")
            results, aggregated, overall = run_one_scenario(
                all_measures, interp_table, renovation_data, assumptions
            )
            write_per_variant_csv(tariff_dir, year, results, all_ids)
            write_aggregated_csv(tariff_dir, year, aggregated, overall, all_ids)

            ert_rows.append({
                'year': year,
                'tariff': tariff_label,
                'kind': tariff['kind'],
                'level': tariff['level'] if tariff['level'] is not None else 0.0,
                'expected_dRT_K': expected_rt_reduction(aggregated, all_ids),
            })

            if tariff['kind'] in ('linear', 'stepped'):
                level_col = 'slope' if tariff['kind'] == 'linear' else 'rate'
                rows = sweep_linear_rows if tariff['kind'] == 'linear' else sweep_stepped_rows
                for mid in all_ids:
                    avg_cost = float(np.mean([
                        agg['weighted_costs'][mid] for agg in aggregated.values()
                    ])) if aggregated else 0.0
                    rows.append({
                        'year': year,
                        level_col: tariff['level'],
                        'measure_id': mid,
                        'measure': MEASURE_SHORT_NAMES.get(mid, mid),
                        'stock_weighted_share': overall[mid],
                        'avg_weighted_cost': avg_cost,
                    })

    sweep_dir = measure_set_root / 'sweep_summary'
    sweep_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (('sweep_linear.csv', sweep_linear_rows),
                       ('sweep_stepped.csv', sweep_stepped_rows)):
        pd.DataFrame(rows).to_csv(sweep_dir / name, index=False)
        print(f"    [OK] {sweep_dir / name}")

    ert_csv = measure_set_root / 'expected_rt_reduction.csv'
    pd.DataFrame(ert_rows).to_csv(ert_csv, index=False)
    print(f"    [OK] {ert_csv}")

    elapsed = time.time() - t_start
    print(f"\n[DONE] choice set '{measure_set_root.name}' in {elapsed:.1f}s")
    return baseline_rts


def main():
    # Every run writes the same file names, so existing results are simply
    # overwritten. (Deleting the folder first fails on synced drives such as
    # OneDrive, which keep directory handles open.)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    log_file = open(OUTPUT_ROOT / 'run.log', 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)
    try:
        t0 = time.time()
        baseline_rts = None
        for ms in MEASURE_SETS:
            root = OUTPUT_ROOT / ms['name']
            root.mkdir(parents=True, exist_ok=True)
            baseline_rts = run_for_measure_set(root, ms['excluded'],
                                               baseline_rts_cache=baseline_rts)
        print(f"\nALL DONE in {time.time() - t0:.1f}s -- outputs in {OUTPUT_ROOT}")
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == '__main__':
    main()
