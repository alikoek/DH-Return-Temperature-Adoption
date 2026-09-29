"""
Step 3 of 5: network value of the return-temperature reduction.

Puts the network side of the owner-network mismatch in EUR, next to the
owner's side, using the published cost-reduction gradient (CRG, paper Eq. 3):

    V_net,i,v = CRG * dT_RT,i,v * E_base,v

per measure i and building variant v, stock-weighted with the number of
represented buildings, for the 2030 reference case. dT_RT,i,v is the
interpolated return-temperature reduction used in the choice model and
E_base,v the variant's final energy demand (MWh/a).

Also reports, per measure, the owner's net annual cost change versus
do-nothing (stock-weighted total cost difference) and the transfer a
motivational tariff would make for the same measure (Danish 1 %/K linear
rule and the +-5 % stepped scheme), so that all three can be compared
(paper Table 4 and Section 5.3).

CRG values (EUR MWh^-1 K^-1, per kelvin of RETURN temperature):
  0.12  mean of 27 Swedish networks (Guelpa et al. 2023), range 0.04-0.38
  0.10  flue-gas condensation, 0.11 reduced pumping, 0.55 network capacity,
        0.58 waste heat, 0.67 geothermal (Geyer et al. 2021, Austrian data;
        the heat-pump and CHP cases refer to the supply temperature and are
        not used)
The paper reports 0.12 and 0.67; the 0.10 column is kept for reference.

Output: outputs/paper_analysis/base_case/network_value_2030.csv

Run:  python compute_network_value.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from load_data import MEASURE_SHORT_NAMES, load_building_baseline_rt
from load_invert_data import load_renovation_data
from run_paper_analysis import setup_measures, run_base_for_year

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'paper_analysis' / 'base_case'
YEAR = 2030
CRG_LOW, CRG_MID, CRG_HIGH = 0.10, 0.12, 0.67   # EUR / (MWh K), return-specific
LINEAR_SLOPE = 1.0                               # EUR / (MWh K), Danish 1 %/K rule
STEPPED_RATE = 0.05                              # +-5 % Woergl / St. Johann
STEP_LOW, STEP_HIGH = 35.0, 39.0                 # neutral band (deg C)


def stepped_factor(t_ret):
    if t_ret < STEP_LOW:
        return -STEPPED_RATE
    if t_ret > STEP_HIGH:
        return +STEPPED_RATE
    return 0.0


def main():
    all_measures, all_ids, _ = setup_measures()
    renovation = load_renovation_data(year=YEAR)
    all_results, _, _ = run_base_for_year(all_measures, renovation)
    # Workbook names differ in spacing ('1960_TOP12' vs '1960_TOP 12'); key by
    # the space-free form.
    base_rt = {k.replace(' ', ''): v for k, v in load_building_baseline_rt().items()}

    rows = []
    for (ait_name, action), r in sorted(all_results.items()):
        w = r['number_of_buildings']
        e_mwh = r['fed_sh_per_bssh'] / 1000.0
        t_base = base_rt[ait_name.replace(' ', '')]
        c0 = r['total_costs']['baseline']
        e_cost0 = r['cost_details']['baseline']['energy_cost']
        for mid in all_ids:
            if mid == 'baseline':
                d_rt = 0.0
            else:
                d_rt = r['interpolated_effects'][mid]['rt_reduction']
            e_cost = r['cost_details'][mid]['energy_cost']
            t_ret = t_base - d_rt
            rows.append({
                'building_class': ait_name, 'action': action, 'weight': w,
                'measure_id': mid, 'd_rt': d_rt, 'e_mwh': e_mwh,
                'net_cost_change': r['total_costs'][mid] - c0,
                'v_low': CRG_LOW * d_rt * e_mwh,
                'v_mid': CRG_MID * d_rt * e_mwh,
                'v_high': CRG_HIGH * d_rt * e_mwh,
                # tariff transfers relative to do-nothing at the same building
                'linear_transfer': LINEAR_SLOPE * d_rt * e_mwh,
                'stepped_transfer': -(stepped_factor(t_ret) * e_cost
                                      - stepped_factor(t_base) * e_cost0),
            })
    df = pd.DataFrame(rows)

    def wavg(g, col):
        return np.average(g[col], weights=g['weight'])

    out = []
    for mid in all_ids:
        g = df[df['measure_id'] == mid]
        out.append({
            'measure_id': mid,
            'measure': MEASURE_SHORT_NAMES.get(mid, mid),
            'd_rt_K': wavg(g, 'd_rt'),
            'heat_MWh': wavg(g, 'e_mwh'),
            'net_cost_change_EUR': wavg(g, 'net_cost_change'),
            'v_net_low_EUR': wavg(g, 'v_low'),
            'v_net_mid_EUR': wavg(g, 'v_mid'),
            'v_net_high_EUR': wavg(g, 'v_high'),
            'linear_transfer_EUR': wavg(g, 'linear_transfer'),
            'stepped_transfer_EUR': wavg(g, 'stepped_transfer'),
        })
    res = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'network_value_{YEAR}.csv'
    res.to_csv(path, index=False)
    pd.set_option('display.width', 200)
    print(res.round(1).to_string(index=False))
    print(f"\n[OK] {path}")


if __name__ == '__main__':
    main()
