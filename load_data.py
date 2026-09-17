"""
Measure data loader.

Reads investment costs, operating costs, lifetimes and the simulated effects
(supply- and return-temperature reduction, efficiency improvement) of the nine
building-side measures for the nine building archetypes from
data/building_measures.xlsx.

Measures (IDs as used in the workbook):
- 1.3:  M1 Flow limiter adjustment
- 1.5:  M2 Heat exchanger renovation
- 2.1:  M3 Hydraulic balancing
- 2.3:  M4 Pump downsizing
- 2.5:  M5 Buffer storage (literature-based investment costs, see
        FALLBACK_INVESTMENT_COSTS)
- 2.7:  M6 User behavior improvement
- 2.8:  M7 Pump downsizing + flow limiter
- 2.9:  M8 Pump downsizing + flow limiter + hydraulic balancing
- 2.10: M9 Hydraulic balancing + heat exchanger

Investment costs are absolute EUR per building; the efficiency improvement is
the reduction of final energy demand in percent of the pre-measure demand.
"""
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / 'data'
DEFAULT_EXCEL_FILE = str(DATA_DIR / 'building_measures.xlsx')

# The nine measures, all loaded directly from the workbook
MEASURE_IDS = ['1.3', '1.5', '2.1', '2.3', '2.5', '2.7', '2.8', '2.9', '2.10']
INDIVIDUAL_MEASURE_IDS = MEASURE_IDS

# Short measure names for visualizations
MEASURE_SHORT_NAMES = {
    '1.3': 'M1: Flow Limiter',
    '1.5': 'M2: Heat Exchanger',
    '2.1': 'M3: Hydraulic Balancing',
    '2.3': 'M4: Pump Downsizing',
    '2.5': 'M5: Buffer Storage',
    '2.7': 'M6: User Behavior',
    '2.8': 'M7: Pump + Flow Limiter',
    '2.9': 'M8: Pump + Flow + Hydr. Bal.',
    '2.10': 'M9: Hydr. Bal. + Heat Exch.',
    'baseline': 'Baseline (Do Nothing)'
}

# Name-based patterns for finding combo measure columns in the Excel file.
# Combo measures don't have numeric IDs in row 2; they are identified by names in row 3.
# Each pattern is a tuple of (must_contain, must_not_contain) substrings.
MEASURE_NAME_PATTERNS = {
    '2.8': {
        'must_contain': ['rderpumpe', 'Mengenbegrenzer'],
        'must_not_contain': ['hydraul'],
    },
    '2.9': {
        'must_contain': ['rderpumpe', 'Mengenbegrenzer', 'hydraul'],
        'must_not_contain': [],
    },
    '2.10': {
        'must_contain': ['hydraulisch', 'rmetauscher'],
        'must_not_contain': [],
    },
}

# Literature-based costs for Pufferspeicher (buffer storage)
# Sources: heizung.de, kesselheld.de, energie-experten.org
# Values include device + installation, scaled by building size
# Measure 2.5 has no investment costs in the file, so we use these fallback values
FALLBACK_INVESTMENT_COSTS = {
    '2.5': {
        '1919_TOP 6': 3000,   # TOP 6: ~750-1000L tank
        '1960_TOP 6': 3000,
        '2001_TOP 6': 3000,
        '1919_TOP 12': 4500,  # TOP 12: ~1500-2000L tank
        '1960_TOP12': 4500,   # Note: typo in the measure workbook (no space)
        '2001_TOP 12': 4500,
        '1919_TOP 24': 7000,  # TOP 24: ~3000-4000L tank
        '1960_TOP 24': 7000,
        '2001_TOP 24': 7000,
    }
}

# Sheet names in the Excel file
SHEET_NAMES = {
    'investment': 'Kosten_1-Investitionen',
    'operational': 'Kosten_2-laufende Kosten',
    'lifetime': 'Kosten_3-Lebensdauern',
    'vt_reduction': 'Effekte_1a-VT-Red',
    'rt_reduction': 'Effekte_2a-RT-Red',
    'rt_absolute': 'Effekte_2b-RT',
    'efficiency': 'Effekte_3-Effizienz'
}


def building_type_matches(full_name, short_name):
    """
    Check if a building type name matches, handling prefixes like '00_basecase___'.

    The RT and VT sheets have building names like '00_basecase___1919_TOP 6',
    while the investment sheet has '1919_TOP 6'.

    Parameters:
    -----------
    full_name : str
        The full building name (potentially with prefix)
    short_name : str
        The short building name to match

    Returns:
    --------
    bool : True if they match
    """
    if full_name is None or short_name is None:
        return False

    full_name = str(full_name).strip()
    short_name = str(short_name).strip()

    # Direct match
    if full_name == short_name:
        return True

    # Check if short_name is at the end of full_name (after prefix)
    if full_name.endswith(short_name):
        return True

    # Check if short_name is contained in full_name
    if short_name in full_name:
        return True

    return False


def find_column_by_measure_id(df, measure_id, search_rows=(0, 1, 2)):
    """
    Find column index by searching for measure ID in header rows.

    The measure workbook has measure IDs (like '1.3', '1.5', etc.) in header rows.
    This function finds the correct column for a given measure ID.

    For combo measures (2.8, 2.9, 2.10) that don't have numeric IDs in header
    rows, falls back to name-based search in row 3 using MEASURE_NAME_PATTERNS.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to search in
    measure_id : str
        The measure ID to find (e.g., '1.3', '2.5', '2.9')
    search_rows : tuple
        Row indices to search for the measure ID

    Returns:
    --------
    int or None : Column index if found, None otherwise
    """
    # First: try exact ID match in header rows
    for row_idx in search_rows:
        for col_idx in range(len(df.columns)):
            try:
                val = df.iloc[row_idx, col_idx]
                # Convert to string and compare
                if str(val).strip() == str(measure_id).strip():
                    return col_idx
            except (IndexError, KeyError):
                continue

    # Second: for combo measures, try name-based search in row 3
    if measure_id in MEASURE_NAME_PATTERNS:
        pattern = MEASURE_NAME_PATTERNS[measure_id]
        must_contain = pattern['must_contain']
        must_not_contain = pattern['must_not_contain']

        # Search row 3 (measure name row) for matching patterns
        name_row = 2  # 0-indexed row 3
        for col_idx in range(len(df.columns)):
            try:
                val = str(df.iloc[name_row, col_idx]).strip()
                if pd.isna(df.iloc[name_row, col_idx]) or val == 'nan':
                    continue

                # Check all must_contain patterns
                if all(substr in val for substr in must_contain):
                    # Check no must_not_contain patterns
                    if not any(substr in val for substr in must_not_contain):
                        return col_idx
            except (IndexError, KeyError):
                continue

    return None


def load_measure_data(excel_file, measure_id='1.3'):
    """
    Load data for one measure from the Excel file.

    Parameters:
    -----------
    excel_file : str
        Path to the Excel file
    measure_id : str
        Measure ID to load (e.g., '1.3', '1.5', '2.1', '2.3', '2.5', '2.7', '2.8', '2.9', '2.10')

    Returns:
    --------
    dict : Dictionary containing measure data with keys:
        - 'building_types': List of building type names
        - 'investment_costs': Investment costs in EUR (absolute)
        - 'operational_costs': Operational costs in EUR/year
        - 'lifetimes': Lifetime in years
        - 'vt_reduction': Supply temperature reduction in K
        - 'rt_reduction': Return temperature reduction in K
        - 'efficiency_improvement': Efficiency improvement in %
        - 'measure_name': Name of the measure
    """

    print("=" * 80)
    print(f"LOADING MEASURE DATA: {measure_id}")
    print("=" * 80)

    # Read investment costs sheet to get building types and find column
    print("\n1. Reading investment costs...")
    investments = pd.read_excel(excel_file, sheet_name=SHEET_NAMES['investment'], header=None)

    # Find the column for this measure
    measure_col = find_column_by_measure_id(investments, measure_id)

    if measure_col is None:
        print(f"   [WARNING] Could not find column for measure {measure_id} in investment sheet")
        # For measure 2.5, this is expected - we'll use fallback costs
        if measure_id == '2.5':
            print(f"   [INFO] Using fallback literature-based costs for measure 2.5")
        measure_col = -1  # Flag for missing column
    else:
        print(f"   Found measure {measure_id} in column {measure_col} ({chr(65 + measure_col)})")

    # Get measure name from row 2 (if available)
    measure_name = None
    if measure_col >= 0:
        try:
            measure_name = investments.iloc[2, measure_col]
            if pd.isna(measure_name) or not isinstance(measure_name, str):
                measure_name = None
        except:
            pass

    if measure_name is None:
        measure_name = f"Measure {measure_id}"

    print(f"   Measure name: {measure_name}")

    # Get building types and investment costs
    # Data starts at row 3 (0-indexed), building types in column 0 or 1
    building_types = []
    investment_costs_list = []

    # Get actual row count from dataframe
    max_rows = len(investments)

    # Scan rows to find building data
    for row_idx in range(3, min(15, max_rows)):  # Extended range but limited to actual rows
        # Get building type from column 0 or 1
        building_type = None
        if pd.notna(investments.iloc[row_idx, 0]) and isinstance(investments.iloc[row_idx, 0], str):
            building_type = investments.iloc[row_idx, 0].strip()
        elif pd.notna(investments.iloc[row_idx, 1]) and isinstance(investments.iloc[row_idx, 1], str):
            building_type = investments.iloc[row_idx, 1].strip()

        # Skip if no building type
        if not building_type or building_type == "" or "Unnamed" in str(building_type):
            continue

        # Get investment cost
        inv_cost = None
        if measure_col >= 0:
            try:
                inv_cost = investments.iloc[row_idx, measure_col]
                if pd.notna(inv_cost) and isinstance(inv_cost, (int, float)):
                    inv_cost = float(inv_cost)
                else:
                    inv_cost = None
            except:
                pass

        # Use fallback costs if available
        if inv_cost is None and measure_id in FALLBACK_INVESTMENT_COSTS:
            fallback_costs = FALLBACK_INVESTMENT_COSTS[measure_id]
            if building_type in fallback_costs:
                inv_cost = fallback_costs[building_type]
                print(f"   Using fallback cost for {building_type}: {inv_cost} EUR")

        # Only add if we have a valid investment cost
        if inv_cost is not None:
            building_types.append(building_type)
            investment_costs_list.append(inv_cost)

    n_buildings = len(building_types)
    print(f"   Found {n_buildings} building types with cost data")

    if n_buildings == 0:
        raise ValueError(f"No building data found for measure {measure_id}")

    # Read operational costs
    print("\n2. Reading operational costs...")
    op_costs_df = pd.read_excel(excel_file, sheet_name=SHEET_NAMES['operational'], header=None)
    op_col = find_column_by_measure_id(op_costs_df, measure_id)
    operational_costs_list = []

    max_op_rows = len(op_costs_df)
    for i, building_type in enumerate(building_types):
        op_cost = 0.0  # Default to 0
        if op_col is not None and op_col >= 0:
            # Find the row for this building type
            for row_idx in range(3, min(15, max_op_rows)):
                bt = None
                if pd.notna(op_costs_df.iloc[row_idx, 0]) and isinstance(op_costs_df.iloc[row_idx, 0], str):
                    bt = op_costs_df.iloc[row_idx, 0].strip()
                elif pd.notna(op_costs_df.iloc[row_idx, 1]) and isinstance(op_costs_df.iloc[row_idx, 1], str):
                    bt = op_costs_df.iloc[row_idx, 1].strip()

                if bt == building_type:
                    try:
                        val = op_costs_df.iloc[row_idx, op_col]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            op_cost = float(val)
                    except:
                        pass
                    break
        operational_costs_list.append(op_cost)

    print(f"   Operational costs loaded for {len(operational_costs_list)} buildings")

    # Read lifetimes
    print("\n3. Reading lifetimes...")
    lifetimes_df = pd.read_excel(excel_file, sheet_name=SHEET_NAMES['lifetime'], header=None)
    lt_col = find_column_by_measure_id(lifetimes_df, measure_id)
    lifetimes_list = []
    max_lt_rows = len(lifetimes_df)

    for i, building_type in enumerate(building_types):
        lifetime = 20.0  # Default to 20 years
        if lt_col is not None and lt_col >= 0:
            for row_idx in range(3, min(15, max_lt_rows)):
                bt = None
                if pd.notna(lifetimes_df.iloc[row_idx, 0]) and isinstance(lifetimes_df.iloc[row_idx, 0], str):
                    bt = lifetimes_df.iloc[row_idx, 0].strip()
                elif pd.notna(lifetimes_df.iloc[row_idx, 1]) and isinstance(lifetimes_df.iloc[row_idx, 1], str):
                    bt = lifetimes_df.iloc[row_idx, 1].strip()

                if bt == building_type:
                    try:
                        val = lifetimes_df.iloc[row_idx, lt_col]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            lifetime = float(val)
                    except:
                        pass
                    break
        lifetimes_list.append(lifetime)

    print(f"   Lifetimes loaded for {len(lifetimes_list)} buildings")

    # Read supply temperature reduction (VT-Red)
    print("\n4. Reading supply temperature reduction...")
    vt_red_df = pd.read_excel(excel_file, sheet_name=SHEET_NAMES['vt_reduction'], header=None)
    vt_col = find_column_by_measure_id(vt_red_df, measure_id)
    vt_reduction_list = []
    max_vt_rows = len(vt_red_df)

    for i, building_type in enumerate(building_types):
        vt_red = 0.0  # Default to 0
        if vt_col is not None and vt_col >= 0:
            for row_idx in range(3, min(35, max_vt_rows)):  # Extended range for effects sheets
                bt = None
                # Effects sheets have building type in column 1 with prefix
                if pd.notna(vt_red_df.iloc[row_idx, 1]) and isinstance(vt_red_df.iloc[row_idx, 1], str):
                    bt = vt_red_df.iloc[row_idx, 1].strip()
                elif pd.notna(vt_red_df.iloc[row_idx, 0]) and isinstance(vt_red_df.iloc[row_idx, 0], str):
                    bt = vt_red_df.iloc[row_idx, 0].strip()

                if building_type_matches(bt, building_type):
                    try:
                        val = vt_red_df.iloc[row_idx, vt_col]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            vt_red = float(val)
                    except:
                        pass
                    break
        vt_reduction_list.append(vt_red)

    print(f"   VT reduction loaded for {len(vt_reduction_list)} buildings")

    # Read return temperature reduction (RT-Red)
    # NOTE: RT sheet has different column positions!
    print("\n5. Reading return temperature reduction...")
    rt_red_df = pd.read_excel(excel_file, sheet_name=SHEET_NAMES['rt_reduction'], header=None)
    rt_col = find_column_by_measure_id(rt_red_df, measure_id)

    if rt_col is not None:
        print(f"   Found RT column for {measure_id} at index {rt_col} ({chr(65 + rt_col)})")
    else:
        print(f"   [WARNING] RT column not found for {measure_id}, using default 0")

    rt_reduction_list = []
    max_rt_rows = len(rt_red_df)

    for i, building_type in enumerate(building_types):
        rt_red = 0.0  # Default to 0
        if rt_col is not None and rt_col >= 0:
            for row_idx in range(3, min(35, max_rt_rows)):  # Extended range for effects sheets
                bt = None
                # Effects sheets have building type in column 1 with prefix
                if pd.notna(rt_red_df.iloc[row_idx, 1]) and isinstance(rt_red_df.iloc[row_idx, 1], str):
                    bt = rt_red_df.iloc[row_idx, 1].strip()
                elif pd.notna(rt_red_df.iloc[row_idx, 0]) and isinstance(rt_red_df.iloc[row_idx, 0], str):
                    bt = rt_red_df.iloc[row_idx, 0].strip()

                if building_type_matches(bt, building_type):
                    try:
                        val = rt_red_df.iloc[row_idx, rt_col]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            rt_red = float(val)
                    except:
                        pass
                    break
        rt_reduction_list.append(rt_red)

    print(f"   RT reduction loaded for {len(rt_reduction_list)} buildings")

    # Read efficiency improvement
    print("\n6. Reading efficiency improvement...")
    efficiency_df = pd.read_excel(excel_file, sheet_name=SHEET_NAMES['efficiency'], header=None)
    eff_col = find_column_by_measure_id(efficiency_df, measure_id)
    efficiency_improvement_list = []
    max_eff_rows = len(efficiency_df)

    for i, building_type in enumerate(building_types):
        efficiency = 0.0  # Default to 0
        if eff_col is not None and eff_col >= 0:
            for row_idx in range(3, min(15, max_eff_rows)):
                bt = None
                if pd.notna(efficiency_df.iloc[row_idx, 0]) and isinstance(efficiency_df.iloc[row_idx, 0], str):
                    bt = efficiency_df.iloc[row_idx, 0].strip()
                elif pd.notna(efficiency_df.iloc[row_idx, 1]) and isinstance(efficiency_df.iloc[row_idx, 1], str):
                    bt = efficiency_df.iloc[row_idx, 1].strip()

                if bt == building_type:
                    try:
                        val = efficiency_df.iloc[row_idx, eff_col]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            efficiency = float(val)
                    except:
                        pass
                    break
        efficiency_improvement_list.append(efficiency)

    print(f"   Efficiency improvement loaded for {len(efficiency_improvement_list)} buildings")

    # Compile all data into a dictionary
    measure_data = {
        'measure_id': measure_id,
        'measure_name': measure_name,
        'building_types': building_types,
        'investment_costs': np.array(investment_costs_list),
        'operational_costs': np.array(operational_costs_list),
        'lifetimes': np.array(lifetimes_list),
        'vt_reduction': np.array(vt_reduction_list),
        'rt_reduction': np.array(rt_reduction_list),
        'efficiency_improvement': np.array(efficiency_improvement_list)
    }

    print("\n" + "=" * 80)
    print("DATA LOADING COMPLETE")
    print("=" * 80)
    print(f"\nLoaded data for: {measure_name} ({measure_id})")
    print(f"Number of building types: {len(building_types)}")
    print(f"\nBuilding types:")
    for i, bt in enumerate(building_types):
        print(f"  {i+1}. {bt}")

    return measure_data


def load_building_baseline_rt(excel_file=None, reference_measure_id='1.3'):
    """
    Load building-specific baseline return temperatures from Effekte_2b-RT.

    The Effekte_2b-RT sheet contains absolute RT values. For measures 1.3-2.3
    (same simulation methodology), the values are identical per building and
    represent the building's baseline return temperature (annual mean).

    We use measure 1.3 as reference (any of 1.3-2.3 would give the same result).

    NOTE: Measures 2.5 and 2.7 use different simulation software and produce
    different absolute RT values (~31.5°C). These are NOT used for baseline RT.

    Parameters:
    -----------
    excel_file : str, optional
        Path to the measure workbook. If None, uses DEFAULT_EXCEL_FILE.
    reference_measure_id : str, optional
        Measure ID to use for reading baseline RT. Default '1.3'.

    Returns:
    --------
    dict : Mapping of building_type -> baseline_rt (°C)
           Example: {'1919_TOP 6': 38.14, '1960_TOP 6': 39.43, ...}
    """
    if excel_file is None:
        excel_file = DEFAULT_EXCEL_FILE

    print("\n" + "=" * 80)
    print("LOADING BUILDING BASELINE RETURN TEMPERATURES")
    print("=" * 80)
    print(f"  Source: Effekte_2b-RT sheet (absolute RT values)")
    print(f"  Reference measure: {reference_measure_id}")

    rt_abs_df = pd.read_excel(excel_file, sheet_name=SHEET_NAMES['rt_absolute'], header=None)

    # Find column for reference measure
    ref_col = find_column_by_measure_id(rt_abs_df, reference_measure_id)
    if ref_col is None:
        print(f"  [WARNING] Could not find column for measure {reference_measure_id}")
        print(f"  [WARNING] Returning empty dict — will fall back to default baseline RT")
        return {}

    print(f"  Found reference column at index {ref_col}")

    # Extract building names and baseline RT values
    baseline_rts = {}
    for row_idx in range(3, min(15, len(rt_abs_df))):
        # Building name is in column 0
        bt = None
        if pd.notna(rt_abs_df.iloc[row_idx, 0]) and isinstance(rt_abs_df.iloc[row_idx, 0], str):
            bt = rt_abs_df.iloc[row_idx, 0].strip()

        if bt is None or bt == '':
            continue

        try:
            val = rt_abs_df.iloc[row_idx, ref_col]
            if pd.notna(val) and isinstance(val, (int, float)):
                baseline_rts[bt] = float(val)
        except (IndexError, KeyError):
            continue

    print(f"\n  Loaded baseline RT for {len(baseline_rts)} buildings:")
    for bt, rt in baseline_rts.items():
        print(f"    {bt:<20} {rt:.2f} °C")

    if baseline_rts:
        rts = list(baseline_rts.values())
        print(f"\n  Range: {min(rts):.2f} - {max(rts):.2f} °C  (mean {np.mean(rts):.2f} °C)")

    return baseline_rts


def load_all_measures(excel_file=None, measure_ids=None):
    """
    Load data for ALL measures with complete data.

    All 9 measures (6 individual + 3 combinations) are loaded directly from
    the measure workbook.

    Parameters:
    -----------
    excel_file : str, optional
        Path to the Excel file. If None, uses DEFAULT_EXCEL_FILE.
    measure_ids : list of str, optional
        List of measure IDs to load. If None, uses MEASURE_IDS.

    Returns:
    --------
    dict : Dictionary with measure IDs as keys, each containing measure data:
        {
            '1.3': {measure_data for Mengenbegrenzer},
            '1.5': {measure_data for Waermetauscher},
            '2.1': {measure_data for Hydraulischer Abgleich},
            '2.3': {measure_data for zu grosse Foerderpumpe},
            '2.5': {measure_data for Pufferspeicher},
            '2.7': {measure_data for Nutzerverhalten},
            '2.8': {measure_data for Pump + Flow Limiter},
            '2.9': {measure_data for Pump + Flow + Hydr. Bal.},
            '2.10': {measure_data for Hydr. Bal. + Heat Exch.},
        }
    """

    if excel_file is None:
        excel_file = DEFAULT_EXCEL_FILE

    if measure_ids is None:
        measure_ids = MEASURE_IDS

    print("=" * 80)
    print("LOADING MULTIPLE MEASURES")
    print("=" * 80)
    print(f"\nFile: {excel_file}")
    print(f"Loading {len(measure_ids)} measures: {measure_ids}")
    print()

    all_measures = {}

    for measure_id in measure_ids:
        print(f"\n{'-' * 80}")
        print(f"Loading Measure {measure_id}")
        print(f"{'-' * 80}")

        try:
            measure_data = load_measure_data(excel_file, measure_id=measure_id)

            # Add short name for visualizations
            measure_data['short_name'] = MEASURE_SHORT_NAMES.get(measure_id, f"M{measure_id}")

            all_measures[measure_id] = measure_data

            print(f"[OK] Measure {measure_id} loaded successfully: {measure_data['measure_name']}")

        except Exception as e:
            print(f"[ERROR] Failed to load measure {measure_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("ALL MEASURES LOADED")
    print("=" * 80)
    print(f"\nSuccessfully loaded {len(all_measures)}/{len(measure_ids)} measures")

    if all_measures:
        print("\nMeasures loaded:")
        for measure_id, data in all_measures.items():
            print(f"  - {measure_id}: {data['measure_name']} ({data['short_name']})")

    return all_measures
