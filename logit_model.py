"""
Logit Model Module for DeRiskDH Project
========================================

This module implements the multinomial logit model for predicting technology
adoption (market shares) based on costs.

PEDAGOGICAL NOTE - UNDERSTANDING LOGIT MODELS:
----------------------------------------------

Imagine you're choosing between 3 heating systems:
- System A: 2,000 EUR/year
- System B: 2,500 EUR/year
- System C: 3,000 EUR/year

QUESTION: Will everyone choose System A (cheapest)?

REAL WORLD: No! Because:
1. People have imperfect information
2. Non-cost factors matter (brand loyalty, installer recommendation, etc.)
3. Some people make mistakes or have different preferences

LOGIT MODEL captures this uncertainty!

KEY INSIGHT:
-----------
The logit model says:
- Cheaper options get HIGHER market share (makes sense!)
- But not 100% (realistic!)
- The relationship is probabilistic, not deterministic

MARKET SHARE FORMULA:
--------------------
Market_share[i] = exp(-lambda * relative_cost[i]) / sum(exp(-lambda * relative_cost[j]))

WHERE:
- relative_cost[i] = cost[i] / weighted_average_cost
- lambda = sensitivity parameter

THE MAGIC OF LAMBDA:
-------------------
Lambda controls how sensitive choices are to cost differences:

HIGH LAMBDA (e.g., 12):
  - Very sensitive to costs
  - Big swings in market share from small cost changes
  - Like "perfect economic agents" who only care about money

LOW LAMBDA (e.g., 0.5):
  - Not very sensitive to costs
  - Market shares don't change much with price
  - Like "real people" with habits, preferences, barriers

EXAMPLE WITH NUMBERS:
--------------------
3 alternatives with costs: [2000, 2500, 3000] EUR/year

Lambda = 0.5 (low sensitivity):
  Market shares: [42%, 33%, 25%]  <- Pretty similar!

Lambda = 4 (medium sensitivity):
  Market shares: [65%, 25%, 10%]  <- Clear preference for cheapest

Lambda = 12 (high sensitivity):
  Market shares: [95%, 4%, 1%]    <- Almost everyone picks cheapest!

WHY RELATIVE COSTS?
------------------
From the paper (Equation 8), we use relative_cost = cost[i] / weighted_average_cost

WHY? This prevents irrelevant alternatives from skewing results.

Example: If we add a crazy expensive option (10,000 EUR/year), it shouldn't
affect the choice between reasonable options (2,000 vs 2,500 EUR/year).

Using relative costs keeps focus on relevant differences.
"""
import numpy as np


def calculate_market_shares(
    costs,  # Array of costs for each alternative (EUR/year)
    lambda_param,  # Sensitivity parameter
    initial_shares=None,  # Initial market shares for weighted average (optional)
):
    """
    Calculate market shares using the multinomial logit model with relative penalties.

    This implements Equation (8) from the Invert/EE-Lab paper:

    s[i] = exp(-lambda * r[i]) / sum(exp(-lambda * r[j]))

    where r[i] = cost[i] / weighted_average_cost

    STEP-BY-STEP PROCESS:
    --------------------
    1. Calculate weighted average cost (using market shares as weights)
    2. Calculate relative penalties (each cost / average)
    3. Calculate utilities (exp(-lambda * relative_penalty))
    4. Normalize utilities to get market shares (sum = 100%)

    Parameters:
    -----------
    costs : array-like
        Costs for each alternative (EUR/year)
    lambda_param : float
        Sensitivity parameter (higher = more cost-sensitive)
    initial_shares : array-like, optional
        Initial market shares for calculating weighted average.
        If None, uses equal shares (1/n for each)

    Returns:
    --------
    dict : Dictionary containing:
        - 'market_shares': Array of market shares (fraction, sums to 1.0)
        - 'relative_penalties': Array of relative costs
        - 'utilities': Array of utility values before normalization
        - 'weighted_avg_cost': The weighted average cost used
    """

    costs = np.array(costs, dtype=float)
    n_alternatives = len(costs)

    # Costs are total annual heating costs, so they are always positive and
    # need no shifting before normalization.

    # Step 1: Initialize market shares if not provided
    if initial_shares is None:
        # Start with equal shares
        market_shares = np.ones(n_alternatives) / n_alternatives
    else:
        market_shares = np.array(initial_shares, dtype=float)
        # Normalize to ensure they sum to 1
        market_shares = market_shares / np.sum(market_shares)

    # Step 2: Calculate weighted average cost
    # This uses current market shares as weights
    # WHY? So we're comparing against the "typical" cost in the market
    weighted_avg_cost = np.sum(market_shares * costs)

    # Step 3: Calculate relative penalties
    # Each alternative's cost relative to the market average
    relative_penalties = costs / weighted_avg_cost

    # Step 4: Calculate utilities
    # exp(-lambda * relative_penalty)
    # WHY negative? Higher cost = lower utility = lower market share
    utilities = np.exp(-lambda_param * relative_penalties)

    # Step 5: Calculate market shares (normalize utilities)
    # Each alternative's share = its utility / sum of all utilities
    # This GUARANTEES shares sum to exactly 1.0 (100%)
    new_market_shares = utilities / np.sum(utilities)

    return {
        "market_shares": new_market_shares,
        "relative_penalties": relative_penalties,
        "utilities": utilities,
        "weighted_avg_cost": weighted_avg_cost,
    }


def iterate_market_shares(costs, lambda_param, max_iterations=50, tolerance=1e-6):
    """
    Iteratively calculate market shares until convergence.

    WHY ITERATE?
    -----------
    The market shares depend on the weighted average cost, which depends on
    the market shares! This is circular.

    Solution: Start with equal shares, calculate new shares, repeat until
    the shares stop changing (convergence).

    EXAMPLE:
    -------
    Iteration 0: shares = [0.33, 0.33, 0.33] (equal start)
    Iteration 1: shares = [0.45, 0.35, 0.20] (updated based on costs)
    Iteration 2: shares = [0.46, 0.34, 0.20] (small change)
    Iteration 3: shares = [0.46, 0.34, 0.20] (converged!)

    Parameters:
    -----------
    costs : array-like
        Costs for each alternative
    lambda_param : float
        Sensitivity parameter
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence criterion (max change in shares)

    Returns:
    --------
    dict : Same as calculate_market_shares, plus:
        - 'iterations': Number of iterations until convergence
        - 'converged': Boolean indicating if convergence was reached
    """

    costs = np.array(costs, dtype=float)
    n_alternatives = len(costs)

    # Start with equal market shares
    current_shares = np.ones(n_alternatives) / n_alternatives

    for iteration in range(max_iterations):
        # Calculate new shares based on current shares
        result = calculate_market_shares(costs, lambda_param, current_shares)
        new_shares = result["market_shares"]

        # Check convergence
        max_change = np.max(np.abs(new_shares - current_shares))

        if max_change < tolerance:
            # Converged!
            result["iterations"] = iteration + 1
            result["converged"] = True
            return result

        # Update for next iteration
        current_shares = new_shares

    # Didn't converge within max_iterations
    result["iterations"] = max_iterations
    result["converged"] = False
    print(f"Warning: Did not converge after {max_iterations} iterations")

    return result
