"""
Interpolation of measure effects for renovated buildings.

Measure effects are simulated for the nine unrenovated archetypes only.
For renovated variants, the efficiency improvement, return-temperature
reduction and supply-temperature reduction are linearly interpolated over the
post-renovation specific space-heating demand (hwb_norm), within the same
size category (TOP 6 / 12 / 24), using the 1919, 1960 and 2001 archetypes as
support points. Outside the support range the boundary values are used (no
extrapolation). Investment costs, lifetimes and operating costs do not depend
on the envelope and are kept constant.
"""
import numpy as np


def parse_building_size(building_type):
    """
    Extract the size category from a building type name.

    Examples:
        '1919_TOP 6'  -> 'TOP 6'
        '1960_TOP12'  -> 'TOP 12'  (handles missing space)
        '2001_TOP 24' -> 'TOP 24'

    Parameters:
    -----------
    building_type : str
        Building type name (e.g., '1919_TOP 6')

    Returns:
    --------
    str : Normalized size category ('TOP 6', 'TOP 12', or 'TOP 24')
    """
    bt = str(building_type).strip()

    # Find 'TOP' and extract the number after it
    top_idx = bt.upper().find('TOP')
    if top_idx == -1:
        raise ValueError(f"Cannot parse building size from '{building_type}' — no 'TOP' found")

    # Get everything after 'TOP', strip spaces, extract number
    after_top = bt[top_idx + 3:].strip()

    # Extract leading digits
    digits = ''
    for ch in after_top:
        if ch.isdigit():
            digits += ch
        else:
            break

    if not digits:
        raise ValueError(f"Cannot parse apartment count from '{building_type}'")

    return f"TOP {digits}"


def build_interpolation_table(all_measures, building_hwb_mapping):
    """
    Build the interpolation lookup table from the measure data.

    Groups buildings by size category and creates sorted (HWB, effects) pairs
    for each measure within each size group.

    Parameters:
    -----------
    all_measures : dict
        Output from load_data.load_all_measures().
        Keys are measure IDs ('1.3', '1.5', etc.), values are measure_data dicts.
    building_hwb_mapping : dict
        Mapping of archetype -> hwb_norm (kWh/m²/year).
        This is the INTERPOLATION VARIABLE (envelope quality).

    Returns:
    --------
    dict : Interpolation table with structure:
        {
            'TOP 6': {
                'hwb_sorted': [76.2, 171.5, 182.9],  # sorted ascending (kWh/m²/yr)
                'building_types_sorted': ['2001_TOP 6', '1960_TOP 6', '1919_TOP 6'],
                'measures': {
                    '1.3': {
                        'efficiency_improvement': [eff_2001, eff_1960, eff_1919],
                        'rt_reduction': [rt_2001, rt_1960, rt_1919],
                        'vt_reduction': [vt_2001, vt_1960, vt_1919],
                        'investment_costs': {  # NOT interpolated, stored per building
                            '1919_TOP 6': 1500,
                            '1960_TOP 6': 1400,
                            '2001_TOP 6': 1300,
                        },
                        'lifetimes': {building_type: lifetime, ...},
                        'operational_costs': {building_type: op_cost, ...},
                    },
                    '1.5': {...},
                    ...
                }
            },
            'TOP 12': {...},
            'TOP 24': {...}
        }
    """
    print("\n" + "=" * 80)
    print("BUILDING HWB INTERPOLATION TABLE")
    print("=" * 80)

    # Step 1: Group buildings by size category
    size_groups = {}  # size -> [(building_type, hwb)]

    for building_type, hwb in building_hwb_mapping.items():
        size = parse_building_size(building_type)
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append((building_type, hwb))

    print(f"\nSize categories found: {list(size_groups.keys())}")

    # Step 2: Sort each group by HWB (ascending)
    for size in size_groups:
        size_groups[size].sort(key=lambda x: x[1])

    # Print the groupings
    for size, buildings in size_groups.items():
        print(f"\n  {size}:")
        for bt, hwb in buildings:
            print(f"    {bt:<20} HWB = {hwb:>10.0f} kWh/year")

    # Step 3: Build interpolation data for each size group and measure
    interp_table = {}

    for size, buildings in size_groups.items():
        building_types_sorted = [bt for bt, _ in buildings]
        hwb_sorted = [hwb for _, hwb in buildings]

        interp_table[size] = {
            'hwb_sorted': hwb_sorted,
            'building_types_sorted': building_types_sorted,
            'measures': {}
        }

        # Get first measure to determine which measures are available
        # (skip 'baseline' as it has no real effects)
        for measure_id, measure_data in all_measures.items():
            if measure_id == 'baseline':
                continue

            measure_buildings = measure_data['building_types']
            measure_entry = {
                'efficiency_improvement': [],
                'rt_reduction': [],
                'vt_reduction': [],
                'investment_costs': {},
                'lifetimes': {},
                'operational_costs': {},
            }

            for bt in building_types_sorted:
                # Find this building in the measure data
                idx = _find_building_index(bt, measure_buildings)

                if idx is not None:
                    measure_entry['efficiency_improvement'].append(
                        float(measure_data['efficiency_improvement'][idx])
                    )
                    measure_entry['rt_reduction'].append(
                        float(measure_data['rt_reduction'][idx])
                    )
                    measure_entry['vt_reduction'].append(
                        float(measure_data['vt_reduction'][idx])
                    )
                    measure_entry['investment_costs'][bt] = float(
                        measure_data['investment_costs'][idx]
                    )
                    measure_entry['lifetimes'][bt] = float(
                        measure_data['lifetimes'][idx]
                    )
                    measure_entry['operational_costs'][bt] = float(
                        measure_data['operational_costs'][idx]
                    )
                else:
                    print(f"  [WARNING] Building '{bt}' not found in measure {measure_id}")
                    measure_entry['efficiency_improvement'].append(0.0)
                    measure_entry['rt_reduction'].append(0.0)
                    measure_entry['vt_reduction'].append(0.0)

            interp_table[size]['measures'][measure_id] = measure_entry

    print("\n" + "=" * 80)
    print("INTERPOLATION TABLE BUILT SUCCESSFULLY")
    print("=" * 80)

    return interp_table


def _find_building_index(target_building, building_list):
    """
    Find the index of a building type in a list, handling naming variations.

    Handles cases like '1960_TOP12' vs '1960_TOP 12' (missing space in the measure workbook).
    """
    # Direct match
    for i, bt in enumerate(building_list):
        if bt == target_building:
            return i

    # Normalize and try again (remove spaces around numbers)
    target_normalized = target_building.replace(' ', '')
    for i, bt in enumerate(building_list):
        if bt.replace(' ', '') == target_normalized:
            return i

    # Check if one contains the other
    for i, bt in enumerate(building_list):
        if target_building in bt or bt in target_building:
            return i

    return None


def interpolate_measure_effects(interp_table, building_size, post_renovation_hwb, measure_id,
                                original_building_type):
    """
    Interpolate measure effects for a renovated building based on its post-renovation HWB.

    Parameters:
    -----------
    interp_table : dict
        Output from build_interpolation_table()
    building_size : str
        Size category (e.g., 'TOP 6', 'TOP 12', 'TOP 24')
    post_renovation_hwb : float
        Post-renovation energy demand (kWh/year absolute)
    measure_id : str
        Measure to interpolate (e.g., '1.3', '1.5')
    original_building_type : str
        Original building type before renovation (e.g., '1919_TOP 6').
        Used to look up the CONSTANT parameters (investment costs, lifetimes, O&M).

    Returns:
    --------
    dict : Interpolated measure effects:
        {
            'efficiency_improvement': float,  # interpolated
            'rt_reduction': float,            # interpolated
            'vt_reduction': float,            # interpolated
            'investment_cost': float,         # from original building
            'lifetime': float,               # from original building
            'operational_cost': float,        # from original building
            'interpolation_info': {          # metadata for debugging
                'building_size': str,
                'post_renovation_hwb': float,
                'bracket_low': str,
                'bracket_high': str,
                'weight_high': float,
                'clamped': bool
            }
        }
    """
    if building_size not in interp_table:
        raise ValueError(
            f"Building size '{building_size}' not found in interpolation table. "
            f"Available: {list(interp_table.keys())}"
        )

    size_data = interp_table[building_size]
    hwb_sorted = size_data['hwb_sorted']  # ascending
    bt_sorted = size_data['building_types_sorted']

    if measure_id not in size_data['measures']:
        raise ValueError(
            f"Measure '{measure_id}' not found for size '{building_size}'. "
            f"Available: {list(size_data['measures'].keys())}"
        )

    measure_data = size_data['measures'][measure_id]
    eff_sorted = measure_data['efficiency_improvement']
    rt_sorted = measure_data['rt_reduction']
    vt_sorted = measure_data['vt_reduction']

    n = len(hwb_sorted)
    clamped = False

    # Case 1: HWB at or below lowest data point → clamp to lowest
    if post_renovation_hwb <= hwb_sorted[0]:
        interp_eff = eff_sorted[0]
        interp_rt = rt_sorted[0]
        interp_vt = vt_sorted[0]
        bracket_low = bt_sorted[0]
        bracket_high = bt_sorted[0]
        weight_high = 0.0
        clamped = True

    # Case 2: HWB at or above highest data point → clamp to highest
    elif post_renovation_hwb >= hwb_sorted[-1]:
        interp_eff = eff_sorted[-1]
        interp_rt = rt_sorted[-1]
        interp_vt = vt_sorted[-1]
        bracket_low = bt_sorted[-1]
        bracket_high = bt_sorted[-1]
        weight_high = 0.0
        clamped = True

    # Case 3: HWB falls between two data points → linear interpolation
    else:
        # Find the bracket
        for j in range(n - 1):
            if hwb_sorted[j] <= post_renovation_hwb <= hwb_sorted[j + 1]:
                # Linear interpolation weight
                hwb_low = hwb_sorted[j]
                hwb_high = hwb_sorted[j + 1]
                weight_high = (post_renovation_hwb - hwb_low) / (hwb_high - hwb_low)
                weight_low = 1.0 - weight_high

                interp_eff = eff_sorted[j] * weight_low + eff_sorted[j + 1] * weight_high
                interp_rt = rt_sorted[j] * weight_low + rt_sorted[j + 1] * weight_high
                interp_vt = vt_sorted[j] * weight_low + vt_sorted[j + 1] * weight_high

                bracket_low = bt_sorted[j]
                bracket_high = bt_sorted[j + 1]
                break

    # Get constant parameters from original building type
    inv_cost = measure_data['investment_costs'].get(original_building_type, 0.0)
    lifetime = measure_data['lifetimes'].get(original_building_type, 20.0)
    op_cost = measure_data['operational_costs'].get(original_building_type, 0.0)

    # If original building not found in this measure, try to find by size match
    if inv_cost == 0.0 and original_building_type not in measure_data['investment_costs']:
        # Fall back to any building in the same size group
        for bt in bt_sorted:
            if bt in measure_data['investment_costs']:
                inv_cost = measure_data['investment_costs'][bt]
                lifetime = measure_data['lifetimes'].get(bt, 20.0)
                op_cost = measure_data['operational_costs'].get(bt, 0.0)
                break

    return {
        'efficiency_improvement': interp_eff,
        'rt_reduction': interp_rt,
        'vt_reduction': interp_vt,
        'investment_cost': inv_cost,
        'lifetime': lifetime,
        'operational_cost': op_cost,
        'interpolation_info': {
            'building_size': building_size,
            'post_renovation_hwb': post_renovation_hwb,
            'bracket_low': bracket_low,
            'bracket_high': bracket_high,
            'weight_high': weight_high,
            'clamped': clamped,
        }
    }
