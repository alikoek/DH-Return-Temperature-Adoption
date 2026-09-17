"""
Logit model for one (archetype x renovation variant).

For each variant: interpolate the measure effects to its post-renovation
space-heating demand, compute the total annual heating cost of every
alternative (energy cost, optionally under a motivational tariff, plus the
annualized net investment and operating cost), and solve the multinomial
logit for the choice shares.
"""
import numpy as np

from interpolate_measures import interpolate_measure_effects
from logit_model import iterate_market_shares
from motivational_tariff import (
    calculate_motivational_tariff_linear,
    calculate_motivational_tariff_stepped,
)


def build_base_hwb_mapping(renovation_data):
    """
    Extract base case hwb_norm values for each building class.

    Uses 'no action' variant as the base case. Falls back to 'maintenance'
    if 'no action' is not available.

    Returns:
    --------
    dict : Archetype name -> hwb_norm (kWh/m²/year)
    """
    mapping = {}
    for archetype, class_data in renovation_data['building_classes'].items():
        for v in class_data['variants']:
            if v['action'] == 'no action':
                mapping[archetype] = v['hwb_norm']
                break
        else:
            for v in class_data['variants']:
                if v['action'] == 'maintenance':
                    mapping[archetype] = v['hwb_norm']
                    break
    return mapping


def run_logit_for_variant(all_measures, interp_table, archetype, variant,
                          assumptions, building_size):
    """
    Run the logit model for a single (building_class × renovation_variant).

    Steps:
    1. Interpolate measure effects for this variant's hwb_norm
    2. Build a measure_data dict with interpolated effects + variant's fed_sh_per_bssh
    3. Calculate costs for each measure
    4. Run logit model

    Parameters:
    -----------
    all_measures : dict
        Measure data from load_all_measures()
    interp_table : dict
        From build_interpolation_table()
    archetype : str
        Archetype name (e.g., '1919_TOP 6')
    variant : dict
        One renovation variant: {'action': str, 'hwb_norm': float,
                                  'fed_sh_per_bssh': float, 'number_of_buildings': float}
    assumptions : dict
        Economic assumptions (discount_rate, energy_price, etc.)
    building_size : str
        Size category (e.g., 'TOP 6')

    Returns:
    --------
    dict : {
        'market_shares': dict of measure_id -> share,
        'total_costs': dict of measure_id -> EUR/year,
        'interpolated_effects': dict of measure_id -> {eff%, rt, vt},
    }
    """
    measure_ids = [mid for mid in all_measures.keys() if mid != 'baseline']

    # Interpolate measure effects for each measure
    interpolated = {}
    for mid in measure_ids:
        interpolated[mid] = interpolate_measure_effects(
            interp_table, building_size, variant['hwb_norm'], mid, archetype
        )

    # Build cost arrays for the logit model
    # Order: all measures + baseline
    all_ids = measure_ids + ['baseline']
    costs_array = []

    # Energy demand for cost calculation (absolute kWh/year)
    fed = variant['fed_sh_per_bssh']
    energy_price = assumptions.get('energy_price', 0.10)
    discount_rate = assumptions.get('discount_rate', 0.05)
    subsidy_rate = assumptions.get('subsidy_rate', 0.0)
    tariff_params = assumptions.get('motivational_tariff_params', None)

    cost_details = {}

    for mid in all_ids:
        if mid == 'baseline':
            energy_after = fed
            rt_red = 0.0
            ann_inv = 0.0
            om = 0.0
        else:
            eff = interpolated[mid]
            energy_after = fed * (1.0 - eff['efficiency_improvement'] / 100.0)
            rt_red = eff['rt_reduction']

            # Annualized investment
            net_inv = eff['investment_cost'] * (1.0 - subsidy_rate)
            lifetime = eff['lifetime']
            if discount_rate > 0:
                crf = discount_rate * (1 + discount_rate)**lifetime / ((1 + discount_rate)**lifetime - 1)
            else:
                crf = 1.0 / lifetime
            ann_inv = net_inv * crf
            om = eff['operational_cost']

        # Energy cost (with optional motivational tariff adjustment)
        energy_cost = energy_after * energy_price
        if tariff_params is not None:
            building_baseline_rts = tariff_params.get('building_baseline_rts', {})
            baseline_rt = building_baseline_rts.get(
                archetype, tariff_params.get('baseline_rt', 38.4)
            )
            approach = tariff_params.get('approach', 'linear')
            if approach == 'linear':
                tariff_result = calculate_motivational_tariff_linear(
                    baseline_rt=baseline_rt,
                    rt_reduction=rt_red,
                    energy_consumption=energy_after,
                    base_price=energy_price,
                    bonus_malus_factor=tariff_params.get('bonus_malus_factor', 0.001),
                    reference_rt=tariff_params.get('reference_rt', 37)
                )
            else:  # stepped
                tariff_result = calculate_motivational_tariff_stepped(
                    baseline_rt=baseline_rt,
                    rt_reduction=rt_red,
                    energy_consumption=energy_after,
                    base_price=energy_price,
                    bonus_threshold=tariff_params.get('bonus_threshold', 35),
                    malus_threshold=tariff_params.get('malus_threshold', 39),
                    bonus_rate=tariff_params.get('bonus_rate', 0.05),
                    malus_rate=tariff_params.get('malus_rate', 0.05)
                )
            energy_cost = energy_after * tariff_result['adjusted_price']

        total_cost = energy_cost + ann_inv + om
        cost_details[mid] = {
            'energy_cost': energy_cost,
            'annualized_investment': ann_inv,
            'total_annual_cost': total_cost,
        }
        costs_array.append(total_cost)

    # Run logit model
    lambda_param = assumptions.get('lambda', 4)
    logit_result = iterate_market_shares(np.array(costs_array), lambda_param=lambda_param)

    # Pack results
    market_shares = {}
    total_costs = {}
    for i, mid in enumerate(all_ids):
        market_shares[mid] = logit_result['market_shares'][i]
        total_costs[mid] = costs_array[i]

    return {
        'market_shares': market_shares,
        'total_costs': total_costs,
        'cost_details': cost_details,
        'interpolated_effects': {mid: interpolated[mid] for mid in measure_ids},
    }
