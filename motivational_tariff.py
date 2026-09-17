"""
Motivational return-temperature tariffs.

Two bonus/malus forms are implemented:

1. Linear:  p = p_base + a * (T_ret - T_ref)
   Every kelvin below the reference return temperature lowers the energy
   price by a (EUR/kWh/K); every kelvin above raises it by the same amount.

2. Stepped: a discount of rate b below the bonus threshold, a surcharge of
   rate b above the malus threshold, and no change within the neutral band.

T_ret is the archetype's baseline return temperature minus the reduction
achieved by the measure. Parameter ranges follow documented Danish and
Austrian schemes (see the paper, Section 4.3).
"""
import numpy as np


def calculate_motivational_tariff_linear(
    baseline_rt,           # Baseline return temperature (°C)
    rt_reduction,          # RT reduction from measure (°C)
    energy_consumption,    # Annual energy consumption (kWh/year)
    base_price,           # Base energy price (EUR/kWh)
    bonus_malus_factor,   # a parameter (EUR/kWh/°C)
    reference_rt          # Reference return temperature (°C)
):
    """
    Calculate motivational tariff using LINEAR approach.

    Formula (from Lygnerud et al., 2023):
    p_DH = p_DH_ref + a × (T_ret - T_ret_ref)

    Where:
    - Lower RT than reference → negative adjustment → lower price (bonus)
    - Higher RT than reference → positive adjustment → higher price (malus)

    Parameters:
    -----------
    baseline_rt : float
        Current return temperature without measure (°C)
    rt_reduction : float
        Reduction in return temperature from measure (°C)
    energy_consumption : float
        Annual energy consumption (kWh/year)
    base_price : float
        Base energy price without tariff adjustment (EUR/kWh)
    bonus_malus_factor : float
        Price adjustment per degree (EUR/kWh/°C)
        Typical value: 0.001 (equivalent to 1 EUR/MWh/°C)
    reference_rt : float
        Reference return temperature (°C)
        Default: 37°C (midpoint of 35-39°C neutral band for this study)

    Returns:
    --------
    dict : Dictionary containing:
        - 'actual_rt': Return temperature after measure (°C)
        - 'adjusted_price': Energy price with tariff (EUR/kWh)
        - 'price_adjustment': Change in price (EUR/kWh)
        - 'annual_benefit': Annual savings from tariff (EUR/year)
        - 'percentage_change': Price change as percentage

    Example:
    --------
    >>> result = calculate_motivational_tariff_linear(
    ...     baseline_rt=38.4,
    ...     rt_reduction=3.0,
    ...     energy_consumption=100000,
    ...     base_price=0.10,
    ...     bonus_malus_factor=0.001,
    ...     reference_rt=37
    ... )
    >>> print(f"Actual RT: {result['actual_rt']:.1f} C")
    Actual RT: 35.4 C
    """

    # Calculate actual return temperature after measure
    actual_rt = baseline_rt - rt_reduction

    # Calculate price adjustment (negative = bonus, positive = malus)
    # Note: We want LOWER temps to give LOWER prices, so formula is:
    # adjustment = a × (actual_rt - reference_rt)
    price_adjustment = bonus_malus_factor * (actual_rt - reference_rt)

    # Calculate adjusted energy price
    adjusted_price = base_price + price_adjustment

    # Ensure price doesn't go negative (safety check)
    adjusted_price = max(adjusted_price, 0.0)

    # Calculate annual benefit (negative adjustment = savings)
    # Benefit = reduction in energy cost due to lower price
    annual_benefit = -price_adjustment * energy_consumption

    # Calculate percentage change for reporting
    percentage_change = (price_adjustment / base_price) * 100

    return {
        'actual_rt': actual_rt,
        'adjusted_price': adjusted_price,
        'price_adjustment': price_adjustment,
        'annual_benefit': annual_benefit,
        'percentage_change': percentage_change
    }


def calculate_motivational_tariff_stepped(
    baseline_rt,          # Baseline return temperature (°C)
    rt_reduction,         # RT reduction from measure (°C)
    energy_consumption,   # Annual energy consumption (kWh/year)
    base_price,          # Base energy price (EUR/kWh)
    bonus_threshold=35,  # Temperature below which bonus applies (°C)
    malus_threshold=39,  # Temperature above which malus applies (°C)
    bonus_rate=0.05,     # Discount rate for bonus (e.g., 0.05 = 5%)
    malus_rate=0.05      # Surcharge rate for malus (e.g., 0.05 = 5%)
):
    """
    Calculate motivational tariff using STEPPED approach.

    Thresholds derived from simulated annual mean RTs of Austrian MFH archetypes:
    - RT < 35°C: 5% discount (bonus)
    - 35°C <= RT <= 39°C: standard price (neutral)
    - RT > 39°C: 5% surcharge (malus)

    The 4K neutral band (35-39°C) is comparable to the 5K band used in Austrian
    practice (50-55°C in St. Johann/Wörgl). Building baseline RTs range from
    37.2-39.7°C (annual mean), so baselines fall mostly in neutral with some in malus.

    Parameters:
    -----------
    baseline_rt : float
        Current return temperature without measure (°C)
    rt_reduction : float
        Reduction in return temperature from measure (°C)
    energy_consumption : float
        Annual energy consumption (kWh/year)
    base_price : float
        Base energy price without tariff adjustment (EUR/kWh)
    bonus_threshold : float
        Temperature below which bonus applies (°C), default=35
    malus_threshold : float
        Temperature above which malus applies (°C), default=39
    bonus_rate : float
        Discount rate for bonus (fraction), default=0.05 (5%)
    malus_rate : float
        Surcharge rate for malus (fraction), default=0.05 (5%)

    Returns:
    --------
    dict : Dictionary containing:
        - 'actual_rt': Return temperature after measure (°C)
        - 'adjusted_price': Energy price with tariff (EUR/kWh)
        - 'price_adjustment': Change in price (EUR/kWh)
        - 'annual_benefit': Annual savings from tariff (EUR/year)
        - 'tariff_zone': Which zone ('bonus', 'neutral', 'malus')
        - 'percentage_change': Price change as percentage

    Example:
    --------
    >>> result = calculate_motivational_tariff_stepped(
    ...     baseline_rt=38.4,
    ...     rt_reduction=3.5,
    ...     energy_consumption=100000,
    ...     base_price=0.10
    ... )
    >>> print(f"Zone: {result['tariff_zone']}, Savings: {result['annual_benefit']:.0f} EUR")
    Zone: bonus, Savings: 500 EUR
    """

    # Calculate actual return temperature after measure
    actual_rt = baseline_rt - rt_reduction

    # Determine which tariff zone we're in
    if actual_rt < bonus_threshold:
        # Bonus zone - discount applies
        tariff_zone = 'bonus'
        price_multiplier = 1.0 - bonus_rate
        percentage_change = -bonus_rate * 100

    elif actual_rt <= malus_threshold:
        # Neutral zone - standard price
        tariff_zone = 'neutral'
        price_multiplier = 1.0
        percentage_change = 0.0

    else:
        # Malus zone - surcharge applies
        tariff_zone = 'malus'
        price_multiplier = 1.0 + malus_rate
        percentage_change = malus_rate * 100

    # Calculate adjusted price
    adjusted_price = base_price * price_multiplier

    # Calculate price adjustment (can be negative for bonus)
    price_adjustment = adjusted_price - base_price

    # Calculate annual benefit (negative adjustment = savings)
    annual_benefit = -price_adjustment * energy_consumption

    return {
        'actual_rt': actual_rt,
        'adjusted_price': adjusted_price,
        'price_adjustment': price_adjustment,
        'annual_benefit': annual_benefit,
        'tariff_zone': tariff_zone,
        'percentage_change': percentage_change
    }
