"""
Building-stock loader for the Invert/EE-Lab exchange files.

Each CSV (data/invert_building_stock_<year>.csv) has one row per
(building class x renovation action) with the post-renovation specific
space-heating demand (hwb_norm, kWh/m2/a, used for interpolating measure
effects), the final energy demand (fed_sh_per_bssh, kWh/a, used for costs)
and the number of represented buildings (used as aggregation weight).

Invert building categories map to the archetypes of the measure workbook:
    BCAT_5 -> TOP 6, BCAT_6 -> TOP 12, BCAT_7 -> TOP 24
combined with the construction period (1919, 1960, 2001).
"""
from pathlib import Path

import pandas as pd
import numpy as np
import re

DATA_DIR = Path(__file__).resolve().parent / 'data'
DEFAULT_RENOVATION_FILES = {
    2030: str(DATA_DIR / 'invert_building_stock_2030.csv'),
    2040: str(DATA_DIR / 'invert_building_stock_2040.csv'),
}


def _clean_action_string(action_str):
    """
    Clean the BC_last_action string which may have b'...' bytes prefix.

    Examples:
        "b'renovation 1'" -> "renovation 1"
        "renovation 1"    -> "renovation 1"
        b'renovation 1'   -> "renovation 1"
    """
    s = str(action_str)
    # Remove b'...' wrapper if present
    match = re.match(r"^b'(.*)'$", s)
    if match:
        return match.group(1)
    # Handle actual bytes
    if isinstance(action_str, bytes):
        return action_str.decode('utf-8')
    return s.strip()


def _parse_invert_name(name):
    """
    Parse an Invert building name to extract BCAT number and construction year.

    Handles both old and new naming conventions:
        Old: 'BCAT_5_CP_19191919.0_basecase'
        New: 'DERISC_BSeg_BCAT_5_CP_19191919.0_basecase'

    Returns:
    --------
    dict : {'bcat': int, 'construction_year': int}
    """
    name_str = str(name)

    # Extract BCAT number
    bcat = None
    bcat_match = re.search(r'BCAT_(\d+)', name_str)
    if bcat_match:
        bcat = int(bcat_match.group(1))

    # Extract construction year from CP_YYYYXXXX pattern
    construction_year = None
    cp_match = re.search(r'CP_(\d{4})', name_str)
    if cp_match:
        construction_year = int(cp_match.group(1))

    return {'bcat': bcat, 'construction_year': construction_year}


def load_renovation_data(csv_file=None, year=None, archetype_types=None):
    """
    Load renovation data from CSV format.

    The CSV has one row per (building_class × renovation_action) with columns:
    - name: Invert building name (e.g., 'DERISC_BCAT_5_CP_19191919.0_basecase')
    - BC_last_action: renovation action (e.g., "b'renovation 1'")
    - number_of_buildings: how many buildings undergo this action
    - hwb_norm: post-renovation HWB in kWh/m²/year (INTERPOLATION VARIABLE)
    - fed_sh_per_bssh: post-renovation energy demand in kWh/year (FOR COST CALCULATIONS)

    Parameters:
    -----------
    csv_file : str, optional
        Path to CSV file. If provided, takes precedence over year.
    year : int, optional
        Year (2030 or 2040). If provided and csv_file is None, uses DEFAULT_RENOVATION_FILES[year].
        One of csv_file or year is required.
    archetype_types : list of str, optional
        Archetype names (e.g., ['1919_TOP 6', ...]). Used for mapping.
        If None, mapping is done generically.

    Returns:
    --------
    dict : Renovation data with structure:
        {
            'building_classes': {
                '1919_TOP 6': {
                    'invert_name': 'DERISC_BCAT_5_CP_19191919.0_basecase',
                    'variants': [
                        {
                            'action': 'no action',
                            'hwb_norm': 131.24,
                            'fed_sh_per_bssh': 80596.0,
                            'number_of_buildings': 35.87,
                        },
                        ...
                    ]
                },
                '1960_TOP 6': {...},
                ...
            },
            'all_actions': ['no action', 'maintenance', 'renovation stand', ...],
            'n_building_classes': int,
            'n_variants_per_class': int,
            'year': int or None,
        }
    """
    if csv_file is None:
        if year is not None and year in DEFAULT_RENOVATION_FILES:
            csv_file = DEFAULT_RENOVATION_FILES[year]
        else:
            raise ValueError(
                f"Pass csv_file or a year in {sorted(DEFAULT_RENOVATION_FILES)}")

    print("\n" + "=" * 80)
    print("LOADING RENOVATION DATA FROM CSV")
    print("=" * 80)
    print(f"\nFile: {csv_file}")

    df = pd.read_csv(csv_file, index_col=0)

    print(f"[OK] Loaded {len(df)} rows")

    # Clean the action column
    df['action_clean'] = df['BC_last_action'].apply(_clean_action_string)

    # Get unique building names and actions
    unique_names = df['name'].unique()
    unique_actions = sorted(df['action_clean'].unique())

    print(f"\nBuilding classes: {len(unique_names)}")
    print(f"Renovation actions: {unique_actions}")

    # Parse each unique building name to get BCAT and construction year
    # Then map each name to an archetype
    # Group BCAT_6 buildings to disambiguate TOP 12 vs TOP 24
    parsed_names = {}
    bcat6_by_year = {}  # year -> list of (name, fed for 'no action')

    for name in unique_names:
        info = _parse_invert_name(name)
        parsed_names[name] = info

        if info['bcat'] == 6:
            c_year = info['construction_year']
            # Get the 'no action' fed for disambiguation
            no_action_row = df[(df['name'] == name) & (df['action_clean'] == 'no action')]
            if len(no_action_row) == 0:
                # Try maintenance as fallback
                no_action_row = df[(df['name'] == name) & (df['action_clean'] == 'maintenance')]
            fed = no_action_row['fed_sh_per_bssh'].iloc[0] if len(no_action_row) > 0 else 0
            if c_year not in bcat6_by_year:
                bcat6_by_year[c_year] = []
            bcat6_by_year[c_year].append((name, fed))

    # Detect whether BCAT_7 exists in the dataset
    has_bcat7 = any(info['bcat'] == 7 for info in parsed_names.values())
    if has_bcat7:
        print(f"  BCAT_7 detected -> BCAT_6 maps to TOP 12, BCAT_7 maps to TOP 24")

    # Build name-to-archetype mapping
    name_to_archetype = {}
    for name, info in parsed_names.items():
        bcat = info['bcat']
        c_year = info['construction_year']

        if bcat == 5:
            n_dwellings = 6
        elif bcat == 7:
            n_dwellings = 24
        elif bcat == 6:
            if has_bcat7:
                # New file format: BCAT_7 handles TOP 24, so BCAT_6 is always TOP 12
                n_dwellings = 12
            else:
                # Old file format fallback: disambiguate TOP 12 vs TOP 24
                same_year_entries = bcat6_by_year.get(c_year, [])
                if len(same_year_entries) >= 2:
                    feds = [f for _, f in same_year_entries]
                    my_fed = [f for n, f in same_year_entries if n == name][0]
                    n_dwellings = 24 if my_fed > np.median(feds) else 12
                else:
                    n_dwellings = 12  # Only one BCAT_6 per year → assume TOP 12
        else:
            print(f"  [WARNING] Unknown BCAT_{bcat} for {name}, skipping")
            continue

        archetype = f"{c_year}_TOP {n_dwellings}"
        # Handle workbook typo (1960_TOP12 without space)
        if archetype_types and archetype not in archetype_types:
            alt = f"{c_year}_TOP{n_dwellings}"
            if alt in archetype_types:
                archetype = alt
        name_to_archetype[name] = archetype

    # Build the output structure
    building_classes = {}

    for name in unique_names:
        if name not in name_to_archetype:
            continue

        archetype = name_to_archetype[name]
        rows = df[df['name'] == name].sort_values('action_clean')

        variants = []
        for _, row in rows.iterrows():
            variants.append({
                'action': row['action_clean'],
                'hwb_norm': float(row['hwb_norm']),
                'fed_sh_per_bssh': float(row['fed_sh_per_bssh']),
                'number_of_buildings': float(row['number_of_buildings']),
            })

        building_classes[archetype] = {
            'invert_name': name,
            'variants': variants,
        }

    # Print summary
    print(f"\n{'Archetype':<20} {'Invert Name':<50} {'Variants':>8}")
    print("-" * 80)
    for archetype, data in sorted(building_classes.items()):
        print(f"{archetype:<20} {data['invert_name']:<50} {len(data['variants']):>8}")

    print(f"\n  Variants per building class:")
    for archetype, data in sorted(building_classes.items()):
        print(f"\n  {archetype}:")
        for v in data['variants']:
            print(f"    {v['action']:<20} hwb_norm={v['hwb_norm']:>7.1f}  "
                  f"fed={v['fed_sh_per_bssh']:>10.0f}  "
                  f"n_buildings={v['number_of_buildings']:>6.1f}")

    result = {
        'building_classes': building_classes,
        'all_actions': unique_actions,
        'n_building_classes': len(building_classes),
        'n_variants_per_class': len(unique_actions),
        'year': year,
    }

    print("\n" + "=" * 80)
    print(f"[OK] Loaded {result['n_building_classes']} building classes × "
          f"{result['n_variants_per_class']} renovation actions")
    print("=" * 80)

    return result
