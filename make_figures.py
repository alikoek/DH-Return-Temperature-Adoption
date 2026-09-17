"""
Step 3 of 4: the figures of the paper.

Figures 2-8 and the appendix figures A.1 and B.1 (Figure 1, the framework
diagram, is not data-driven). File names match the figure numbers in the
paper. Every figure reads only the CSVs written by run_scenarios.py and
run_paper_analysis.py (paths are printed for traceability), except Figure 4,
which also reads the measure workbook, and writes PDF and PNG files to
outputs/figures/.

Inputs
------
outputs/scenarios/all_measures/per_tariff/none/        reference case
outputs/scenarios/all_measures/sweep_summary/          tariff sweeps
outputs/scenarios/restricted/per_tariff/none/          restricted choice set
outputs/paper_analysis/                                sensitivities, nested logit

Run:  python make_figures.py
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd

from load_data import MEASURE_SHORT_NAMES, load_all_measures

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SCEN = ROOT / 'outputs' / 'scenarios'
SCEN_NONE = SCEN / 'all_measures' / 'per_tariff' / 'none'
SCEN_SWEEP = SCEN / 'all_measures' / 'sweep_summary'
SCEN_RESTR_NONE = SCEN / 'restricted' / 'per_tariff' / 'none'
PAPER = ROOT / 'outputs' / 'paper_analysis'
FIG_DIR = ROOT / 'outputs' / 'figures'

# -----------------------------------------------------------------------------
# Style: serif, colorblind-safe (Paul Tol muted), no in-figure titles
# -----------------------------------------------------------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

TOL_MUTED = {
    '1.3': '#332288',   # M1 flow limiter      - indigo
    '1.5': '#88CCEE',   # M2 heat exchanger    - cyan
    '2.1': '#44AA99',   # M3 hydraulic bal.    - teal
    '2.3': '#117733',   # M4 pump downsizing   - green
    '2.5': '#999933',   # M5 buffer storage    - olive
    '2.7': '#DDCC77',   # M6 user behavior     - sand
    '2.8': '#CC6677',   # M7 pump+flow         - rose
    '2.9': '#882255',   # M8 pump+flow+bal.    - wine
    '2.10': '#AA4499',  # M9 bal.+HX           - purple
    'baseline': '#777777',
}

ALL_IDS = ['1.3', '1.5', '2.1', '2.3', '2.5', '2.7', '2.8', '2.9', '2.10', 'baseline']
RESTRICTED_IDS = ['1.3', '1.5', '2.1', '2.8', '2.9', '2.10', 'baseline']

TICK_LABELS = {'1.3': 'M1', '1.5': 'M2', '2.1': 'M3', '2.3': 'M4', '2.5': 'M5',
               '2.7': 'M6', '2.8': 'M7', '2.9': 'M8', '2.10': 'M9', 'baseline': 'DN'}

LEGEND_LABELS = {
    '1.3': 'M1 Flow limiter',
    '1.5': 'M2 Heat exchanger',
    '2.1': 'M3 Hydraulic balancing',
    '2.3': 'M4 Pump downsizing',
    '2.5': 'M5 Buffer storage',
    '2.7': 'M6 User behavior',
    '2.8': 'M7 Pump + flow limiter',
    '2.9': 'M8 Pump + flow + balancing',
    '2.10': 'M9 Balancing + heat exch.',
    'baseline': 'Do nothing',
}

NAME_TO_ID = {v: k for k, v in MEASURE_SHORT_NAMES.items()}


def share_col(mid):
    return f"share_{MEASURE_SHORT_NAMES.get(mid, mid)}"


def load_csv(path):
    print(f"  src: {path}")
    return pd.read_csv(path)


def save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = FIG_DIR / f'{stem}.{ext}'
        fig.savefig(out, bbox_inches='tight')
        print(f"  [OK] {out}")
    plt.close(fig)


def measure_legend(fig, ids, extra_handles=(), extra_labels=(), **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, fc=TOL_MUTED[m]) for m in ids]
    labels = [LEGEND_LABELS[m] for m in ids]
    fig.legend(handles + list(extra_handles), labels + list(extra_labels), **kw)


ARCHETYPE_ORDER = ['1919_TOP 6', '1919_TOP 12', '1919_TOP 24',
                   '1960_TOP 6', '1960_TOP12', '1960_TOP 24',
                   '2001_TOP 6', '2001_TOP 12', '2001_TOP 24']

YEAR_HANDLES = [plt.Rectangle((0, 0), 1, 1, fc='#444444'),
                plt.Rectangle((0, 0), 1, 1, fc='#444444', alpha=0.45)]


# -----------------------------------------------------------------------------
# F2: baseline adoption (heatmap + overall bars)
# -----------------------------------------------------------------------------

def make_fig2_baseline_shares():
    print("\nFigure 2: baseline adoption")
    agg30 = load_csv(SCEN_NONE / 'results_aggregated_2030.csv')
    diff = load_csv(PAPER / 'base_case' / 'overall_shares_with_diffusion.csv')

    arche = agg30[agg30['building_class'] != 'OVERALL (stock-weighted)'].copy()
    arche = arche.set_index('building_class').loc[ARCHETYPE_ORDER].reset_index()
    mat = np.array([[arche.iloc[r][share_col(m)] for m in ALL_IDS]
                    for r in range(len(arche))]) * 100

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.46)

    # (a) heatmap: archetype x alternative, base case 2030
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mat, cmap='YlGnBu', aspect='auto', vmin=0)
    ax.set_xticks(range(len(ALL_IDS)), [TICK_LABELS[m] for m in ALL_IDS])
    ax.set_yticks(range(len(arche)),
                  [a.replace('TOP12', 'TOP 12') for a in arche['building_class']])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha='center', va='center',
                    fontsize=7, color='black' if mat[i, j] < 18 else 'white')
    ax.set_xlabel('Alternative')
    ax.grid(False)
    ax.set_title('(a) Choice shares by archetype, 2030 (%)', loc='left')
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    # (b) stock-weighted overall choice shares, 2030 vs 2040
    x = np.arange(len(ALL_IDS))
    ax = fig.add_subplot(gs[0, 1])
    for k, year in enumerate((2030, 2040)):
        d = diff[diff['year'] == year].set_index('measure_id')
        vals = [d.loc[m, 'choice_share'] * 100 for m in ALL_IDS]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38,
               color=[TOL_MUTED[m] for m in ALL_IDS],
               alpha=1.0 if year == 2030 else 0.45,
               edgecolor='white', linewidth=0.3)
    ax.set_xticks(x, [TICK_LABELS[m] for m in ALL_IDS], fontsize=7)
    ax.tick_params(axis='x', pad=1.5)
    ax.set_ylabel('Choice share (%)')
    ax.set_title('(b) Stock-weighted choice shares', loc='left')
    ax.legend(YEAR_HANDLES, ['2030', '2040'], frameon=False)

    save(fig, 'figure2_baseline_adoption')


# -----------------------------------------------------------------------------
# F3: diffusion-adjusted stock adoption
# -----------------------------------------------------------------------------

def make_fig3_stock_adoption():
    """Stock composition and a true zoom on the stock that has decided.

    For each horizon the narrow bar is the whole stock; only the band above the
    gray block has faced a decision occasion. The dashed leaders expand that
    band to full height, so the wide bar carries the same quantities on a
    stretched axis and the individual measures become legible.
    """
    print("\nFigure 3: diffusion-adjusted stock adoption")
    diff = load_csv(PAPER / 'base_case' / 'overall_shares_with_diffusion.csv')
    years = (2030, 2040)
    measures = [m for m in ALL_IDS if m != 'baseline']
    stack = ['baseline'] + measures            # bottom to top, both bars
    panel = {2030: '(a)', 2040: '(b)'}

    fig = plt.figure(figsize=(7.2, 4.2))
    outer = fig.add_gridspec(1, 2, wspace=0.50)

    for c, year in enumerate(years):
        inner = outer[0, c].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=0.20)
        d = diff[diff['year'] == year].set_index('measure_id')
        rate = float(d['diffusion_rate'].iloc[0])
        undecided = (1 - rate) * 100
        stock = {m: float(d.loc[m, 'stock_adoption']) * 100 for m in measures}
        stock['baseline'] = rate * float(d.loc['baseline', 'choice_share']) * 100

        # whole stock
        axL = fig.add_subplot(inner[0, 0])
        axL.bar(0, undecided, 0.6, color='#E4E4E4', edgecolor='white', linewidth=0.4)
        axL.text(0, undecided / 2, f'{undecided:.0f}%\nnot yet\nconsidered\na measure',
                 ha='center', va='center', fontsize=6.5, color='#555555',
                 linespacing=1.4)
        bottom = undecided
        for m in stack:
            axL.bar(0, stock[m], 0.6, bottom=bottom, color=TOL_MUTED[m],
                    edgecolor='white', linewidth=0.4)
            bottom += stock[m]
        axL.set_xticks([0], ['whole stock'], fontsize=7.5)
        axL.set_xlim(-0.7, 0.7)
        axL.set_ylim(0, 100)
        axL.set_yticks([0, 20, 40, 60, 80, 100])
        axL.set_ylabel('Share of stock (%)')
        axL.set_title(f'{panel[year]} {year}', loc='left')

        # the decided band, stretched to full height
        axR = fig.add_subplot(inner[0, 1])
        top = rate * 100
        bottom, centers = 0.0, {}
        for m in stack:
            axR.bar(0, stock[m], 0.5, bottom=bottom, color=TOL_MUTED[m],
                    edgecolor='white', linewidth=0.4)
            centers[m] = bottom + stock[m] / 2
            bottom += stock[m]
        gap = top * 0.048
        ys = _stagger(np.array([centers[m] for m in stack]), gap)
        for m, y_lab in zip(stack, ys):
            axR.plot([0.25, 0.42], [centers[m], y_lab], color='#BBBBBB',
                     lw=0.5, zorder=1, clip_on=False)
            axR.text(0.47, y_lab, f'{TICK_LABELS[m]}  {stock[m]:.1f}', fontsize=7,
                     va='center', ha='left', color=TOL_MUTED[m])
        axR.set_xticks([0], ['considered\na measure'], fontsize=7.5)
        axR.set_xlim(-0.45, 1.5)
        axR.set_ylim(0, top)
        axR.set_yticks(np.arange(0, top + 0.1, 5 if top <= 30 else 10))
        axR.yaxis.tick_right()          # keeps the gap free for the leaders
        axR.spines['right'].set_visible(True)
        axR.tick_params(axis='y', labelsize=7.5)
        axR.grid(axis='x', alpha=0)

        for y_src, y_dst in ((100.0, 1.0), (undecided, 0.0)):
            fig.add_artist(ConnectionPatch(
                xyA=(0.3, y_src), coordsA=axL.transData,
                xyB=(0.0, y_dst), coordsB=axR.transAxes,
                color='#AAAAAA', lw=0.7, ls='--', zorder=6,
                clip_on=False))

    save(fig, 'figure3_stock_adoption')


# -----------------------------------------------------------------------------
# F5: cost decomposition
# -----------------------------------------------------------------------------

def make_fig5_cost_decomposition():
    print("\nFigure 5: cost decomposition")
    cd = load_csv(PAPER / 'base_case' / 'cost_decomposition_2030.csv')
    cd = cd.set_index('measure_id').loc[ALL_IDS].reset_index()
    cd = cd.sort_values('total_cost')

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    y = np.arange(len(cd))
    ax.barh(y, cd['energy_cost'], color='#4477AA', label='Energy cost')
    ax.barh(y, cd['annualized_investment'], left=cd['energy_cost'],
            color='#EE6677', label='Annualized investment')
    base_total = float(cd.loc[cd['measure_id'] == 'baseline', 'total_cost'].iloc[0])
    ax.axvline(base_total, color='#777777', lw=1, ls='--')
    ax.annotate('Do-nothing total', xy=(base_total, 1.0),
                xycoords=('data', 'axes fraction'), xytext=(0, 3),
                textcoords='offset points', fontsize=7, color='#555555',
                ha='center', va='bottom')
    labels = [f"{TICK_LABELS[m]} {LEGEND_LABELS[m].split(' ', 1)[1]}" if m != 'baseline'
              else 'Do nothing' for m in cd['measure_id']]
    ax.set_yticks(y, labels)
    # annotate the annualized-investment component (the economically decisive
    # part) at the end of each bar; the common energy cost dwarfs it visually
    for yi, (tot, inv) in enumerate(zip(cd['total_cost'],
                                        cd['annualized_investment'])):
        if inv >= 1:
            ax.text(tot + 160, yi, f"+{inv:,.0f}", va='center', fontsize=6.5,
                    color='#EE6677')
    ax.set_xlim(0, float(cd['total_cost'].max()) * 1.14)
    ax.set_xlabel('Stock-weighted total annual heating cost per building (EUR/year), 2030')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
    save(fig, 'figure5_cost_decomposition')


# -----------------------------------------------------------------------------
# F6: tariff dose-response
# -----------------------------------------------------------------------------

def _none_anchor(year):
    """Stock-weighted shares at zero tariff intensity (the 'none' run)."""
    agg = pd.read_csv(SCEN_NONE / f'results_aggregated_{year}.csv')
    row = agg[agg['building_class'] == 'OVERALL (stock-weighted)'].iloc[0]
    return {m: float(row[share_col(m)]) for m in ALL_IDS}


# Alternatives whose share moves visibly under the tariff (colored, labeled
# bold); the rest are drawn gray. M4 is kept despite smaller movement because
# it is the negative-dTrt malus exemplar discussed in the text.
DOSE_EMPH = {'1.3', '2.1', '2.9', '2.10',        # gainers (large dTrt)
             '2.3', '2.7', 'baseline'}           # losers


def _stagger(vals, gap):
    """Push label y-positions apart (top-down greedy, preserves order)."""
    order = np.argsort(vals)[::-1]
    out = dict()
    prev = None
    for i in order:
        y = vals[i] if prev is None else min(vals[i], prev - gap)
        out[i] = y
        prev = y
    return [out[i] for i in range(len(vals))]


def _dose_panel(ax, sweep, xcol, xscale, anchor, label_x, gap, lift=0.0):
    """One dose-response panel: delta vs no-tariff anchor (pp), 2030 only."""
    ends = []
    for m in ALL_IDS:
        s = sweep[sweep['measure_id'] == m].sort_values(xcol)
        xs = [0.0] + list(s[xcol] * xscale)
        ys = [0.0] + list((s['stock_weighted_share'] - anchor[m]) * 100)
        emph = m in DOSE_EMPH
        ax.plot(xs, ys, '-', color=TOL_MUTED[m] if emph else '#CCCCCC',
                lw=1.8 if emph else 1.0, marker='o' if emph else None,
                ms=2.8, zorder=3 if emph else 1)
        ends.append(ys[-1])
    ax.axhline(0, color='#888888', lw=0.8, zorder=0)
    for m, y_lab in zip(ALL_IDS, _stagger(np.array(ends), gap)):
        if y_lab < 0:                        # tighten the lower label column
            y_lab += lift
        emph = m in DOSE_EMPH
        ax.text(label_x, y_lab, f"{TICK_LABELS[m]} ({anchor[m] * 100:.1f}%)",
                fontsize=7, va='center', ha='left',
                color=TOL_MUTED[m] if emph else '#999999',
                fontweight='bold' if emph else 'normal')


def make_fig6_tariff_dose_response():
    print("\nFigure 6: tariff dose-response (change vs no tariff, 2030)")
    lin = load_csv(SCEN_SWEEP / 'sweep_linear.csv')
    stp = load_csv(SCEN_SWEEP / 'sweep_stepped.csv')
    print(f"  src: {SCEN_NONE / 'results_aggregated_2030.csv'} (zero anchor)")
    anchor = _none_anchor(2030)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    # (a) linear slope, EUR/(MWh K); slope column is EUR/(kWh K)
    ax = axes[0]
    _dose_panel(ax, lin[lin['year'] == 2030], 'slope', 1000, anchor,
                label_x=1.04, gap=0.105, lift=0.055)
    ax.axvline(1.0, color='#999999', lw=0.8, ls=':', zorder=0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(0, 1.33)
    ax.set_ylim(-0.80, 1.0)
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_xlabel(r'Linear tariff slope $a$ (EUR MWh$^{-1}$ K$^{-1}$)')
    ax.set_ylabel('Change in stock-weighted choice share\nvs. no tariff (pp), 2030')
    ax.set_title('(a) Linear tariff', loc='left')

    # (b) stepped bonus/malus rate, %
    ax = axes[1]
    _dose_panel(ax, stp[stp['year'] == 2030], 'rate', 100, anchor,
                label_x=20.8, gap=0.46)
    ax.axvline(5.0, color='#999999', lw=0.8, ls=':', zorder=0)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(0, 26.5)
    ax.set_ylim(-3.4, 4.6)
    ax.set_xlabel('Stepped bonus/malus rate (%)')
    ax.set_title('(b) Stepped tariff', loc='left')

    fig.tight_layout()
    save(fig, 'figure6_tariff_dose_response')


# -----------------------------------------------------------------------------
# F7: renovation effects
# -----------------------------------------------------------------------------

# Ordered by renovation depth (mean q_SH: 92, 92, 55, 30, 17 kWh/m2a);
# Invert's "renovation 2" is shallower than "renovation stand", so the
# paper labels the packages descriptively: moderate < standard < deep.
ACTION_ORDER = ['no action', 'maintenance', 'renovation 2', 'renovation stand', 'renovation 1']
ACTION_LABELS = ['No action', 'Maintenance', 'Moderate renovation',
                 'Standard renovation', 'Deep renovation']


def make_fig7_renovation_effects():
    print("\nFigure 7: renovation effects")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for panel, (ax, year) in enumerate(zip(axes, (2030, 2040))):
        df = load_csv(SCEN_NONE / f'results_renovation_analysis_{year}.csv')
        x = np.arange(len(ACTION_ORDER))
        bottom = np.zeros(len(ACTION_ORDER))
        for m in ALL_IDS:
            vals = []
            for action in ACTION_ORDER:
                sub = df[df['renovation_action'] == action]
                vals.append(np.average(sub[share_col(m)],
                                       weights=sub['number_of_buildings']) * 100)
            ax.bar(x, vals, 0.62, bottom=bottom, color=TOL_MUTED[m],
                   edgecolor='white', linewidth=0.3)
            if m == 'baseline':   # print the do-nothing share on its segment
                for xi, (b0, v) in enumerate(zip(bottom, vals)):
                    ax.text(xi, b0 + v / 2, f"{v:.0f}", ha='center',
                            va='center', fontsize=6.5, color='white')
            bottom += np.array(vals)
        ax.set_xticks(x, ACTION_LABELS, fontsize=7, rotation=22,
                      ha='right', rotation_mode='anchor')
        ax.set_title(f'({chr(97 + panel)}) {year}', loc='left')
        ax.set_ylim(0, 102)
    axes[0].set_ylabel('Stock-weighted choice share (%)')
    measure_legend(fig, ALL_IDS, loc='lower center', ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, -0.24), fontsize=8)
    save(fig, 'figure7_renovation_effects')


# -----------------------------------------------------------------------------
# F8: sensitivity (lambda, energy price, M6 efficiency)
# -----------------------------------------------------------------------------

def make_fig8_sensitivity_panel():
    print("\nFigure 8: sensitivity panels")
    sens = load_csv(PAPER / 'sensitivity' / 'lambda_price_sensitivity.csv')
    m6 = load_csv(PAPER / 'm6_sensitivity' / 'm6_sensitivity_summary.csv')

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    panels = [
        (axes[0], sens[(sens['param'] == 'lambda') & (sens['year'] == 2030)],
         'value', r'(a) Cost sensitivity $\lambda$', [2, 4, 8]),
        (axes[1], sens[(sens['param'] == 'energy_price') & (sens['year'] == 2030)],
         'value', '(b) Energy price (EUR/kWh)', [0.08, 0.10, 0.12]),
        (axes[2], m6[m6['year'] == 2030],
         'm6_efficiency_pct', '(c) M6 assumed efficiency (%)', [5.0, 9.6, 15.0]),
    ]
    for ax, df, xcol, title, ticks in panels:
        for m in ALL_IDS:
            s = df[df['measure_id'] == m].sort_values(xcol)
            ax.plot(s[xcol], s['stock_weighted_share'] * 100, '-o',
                    color=TOL_MUTED[m], lw=1.3, ms=3)
        ax.set_xticks(ticks)
        ax.set_title(title, loc='left')
        ax.set_xlabel('')
        if xcol == 'm6_efficiency_pct':
            # panel (c) carries the decisive sensitivity: label the two lines
            # whose ranking flips at the 5 % assumption
            for m in ('2.7', '1.3'):
                s = df[df['measure_id'] == m].sort_values(xcol)
                ax.text(float(s[xcol].iloc[-1]) + 0.5,
                        float(s['stock_weighted_share'].iloc[-1]) * 100,
                        TICK_LABELS[m], color=TOL_MUTED[m], fontsize=7.5,
                        fontweight='bold', va='center')
            ax.set_xlim(4, 18)
    axes[0].set_ylabel('Stock-weighted choice share (%)\n2030, reference case')
    measure_legend(fig, ALL_IDS, loc='lower center', ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, -0.18), fontsize=8)
    save(fig, 'figure8_sensitivity')


# -----------------------------------------------------------------------------
# Figure B1 (appendix): full vs restricted measure set
# -----------------------------------------------------------------------------

def make_figB1_measure_set_comparison():
    print("\nFigure B1: measure-set comparison")
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    full = load_csv(SCEN_NONE / 'results_aggregated_2030.csv')
    restr = load_csv(SCEN_RESTR_NONE / 'results_aggregated_2030.csv')
    frow = full[full['building_class'] == 'OVERALL (stock-weighted)'].iloc[0]
    rrow = restr[restr['building_class'] == 'OVERALL (stock-weighted)'].iloc[0]

    x = np.arange(len(ALL_IDS))
    fvals = [float(frow[share_col(m)]) * 100 for m in ALL_IDS]
    rvals = [float(rrow[share_col(m)]) * 100 if m in RESTRICTED_IDS else 0.0
             for m in ALL_IDS]
    ax.bar(x - 0.19, fvals, 0.38, color=[TOL_MUTED[m] for m in ALL_IDS],
           edgecolor='white', linewidth=0.3, label='Full choice set')
    ax.bar(x + 0.19, rvals, 0.38, color=[TOL_MUTED[m] for m in ALL_IDS],
           alpha=0.45, edgecolor='white', linewidth=0.3, hatch='///',
           label='Restricted choice set')
    for xi, m in zip(x, ALL_IDS):
        if m not in RESTRICTED_IDS:
            ax.text(xi + 0.19, 0.6, 'not included', rotation=90, fontsize=6,
                    ha='center', va='bottom', color='#777777')
    ax.set_xticks(x, [TICK_LABELS[m] for m in ALL_IDS])
    ax.set_ylabel('Stock-weighted choice share (%)\n2030, reference case')
    handles = [plt.Rectangle((0, 0), 1, 1, fc='#AAAAAA'),
               plt.Rectangle((0, 0), 1, 1, fc='#AAAAAA', alpha=0.45, hatch='///')]
    ax.legend(handles, ['Full choice set (nine measures)', 'Restricted choice set (M1-M3, M7-M9)'],
              frameon=False)
    save(fig, 'figureB1_measure_set_comparison')


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# A1 (appendix): nested-logit rank robustness (bump chart)
# -----------------------------------------------------------------------------

def make_figA1_nl_rank():
    print("\nFigure A.1: nested-logit rank robustness")
    df = load_csv(PAPER / 'nested_logit' / 'nl_robustness.csv')
    df = df[df['year'] == 2030]
    thetas = [1.0, 0.7, 0.5, 0.3]
    lab = lambda m: 'DN' if m == 'baseline' else TICK_LABELS[m]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True)
    for ax, part, title in zip(axes, ['A', 'B'],
                               ['(a) Choice-structure partition',
                                '(b) Component-family partition']):
        sub = df[df['partition'] == part]
        ranks = {m: [] for m in ALL_IDS}
        for th in thetas:
            s = sub[sub['theta'] == th].set_index('measure_id')['stock_weighted_share']
            r = s.rank(ascending=False, method='min').astype(int)
            for m in ALL_IDS:
                ranks[m].append(int(r[m]))
        x = list(range(len(thetas)))
        for m in ALL_IDS:
            emph = m in ('baseline', '2.7')
            ax.plot(x, ranks[m], '-o', color=TOL_MUTED[m],
                    lw=2.6 if m == 'baseline' else 1.4, ms=5 if emph else 3.5,
                    zorder=5 if emph else 3)
            ax.text(x[-1] + 0.12, ranks[m][-1], lab(m), va='center', ha='left',
                    fontsize=7.5, color=TOL_MUTED[m], fontweight='bold' if emph else 'normal')
            ax.text(x[0] - 0.12, ranks[m][0], lab(m), va='center', ha='right',
                    fontsize=7, color=TOL_MUTED[m])
        ax.set_ylim(10.6, 0.4)
        ax.set_yticks(range(1, 11))
        ax.set_xticks(x, [f'{t:g}' for t in thetas])
        ax.set_xlim(-0.8, len(thetas) - 1 + 0.8)
        ax.set_xlabel(r'$\theta$ ($\leftarrow$ MNL, stronger substitution $\rightarrow$)')
        ax.set_title(title, fontsize=9, loc='left')
        ax.grid(axis='y', alpha=0.25); ax.grid(axis='x', alpha=0)
    axes[0].set_ylabel('Rank among the ten alternatives (1 = most adopted)')
    fig.tight_layout()
    save(fig, 'figureA1_nl_rank')


def make_fig_misalignment():
    """ΔRT (network benefit) vs predicted adoption; bubble = investment (capital).

    The paper's central tension: owners adopt on private cost (energy savings vs
    capital), so the return-temperature benefit the network wants is decoupled
    from what drives uptake.
    """
    print("\nFigure 4: misalignment (delta-RT vs adoption)")
    agg = load_csv(SCEN_NONE / 'results_aggregated_2030.csv')
    row = agg[agg['building_class'] == 'OVERALL (stock-weighted)'].iloc[0]
    # Mean return-temperature reduction and investment across the nine
    # unrenovated archetypes, straight from the measure workbook
    print("  src: data/building_measures.xlsx (archetype means)")
    with contextlib.redirect_stdout(io.StringIO()):
        measure_data = load_all_measures()

    measures = ['1.3', '1.5', '2.1', '2.3', '2.5', '2.7', '2.8', '2.9', '2.10']
    xs, ys, inv = [], [], []
    for mid in measures:
        xs.append(float(np.mean(measure_data[mid]['rt_reduction'])))
        ys.append(float(row[share_col(mid)]) * 100)
        inv.append(float(np.mean(measure_data[mid]['investment_costs'])))
    xs, ys, inv = np.array(xs), np.array(ys), np.array(inv)
    area = 80 + inv / 1000.0 * 45            # bubble area ~ investment

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axvspan(-1.0, 0, color='#f2f2f2', zorder=0)            # measures that raise RT
    ax.axvline(0, color='#999999', lw=0.8, ls='--', zorder=1)
    ax.scatter(xs, ys, s=area, c='#4477AA',
               alpha=0.85, edgecolor='black', linewidth=0.6, zorder=4)

    # label offsets (pts); big bubbles need larger offsets to sit outside
    off = {'1.3': (10, 6), '1.5': (26, -2), '2.1': (15, 7), '2.3': (-15, -2),
           '2.5': (-14, 2), '2.7': (13, 0), '2.8': (16, 2), '2.9': (24, 4),
           '2.10': (15, 17)}
    for m, x, y in zip(measures, xs, ys):
        dx, dy = off[m]
        ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
        va = 'bottom' if dy > 0 else ('top' if dy < 0 else 'center')
        ax.annotate(TICK_LABELS[m], (x, y),
                    textcoords='offset points', xytext=(dx, dy),
                    ha=ha, va=va, fontsize=9.5, fontweight='bold',
                    color='#333333', zorder=6)

    ax.set_xlim(-1.0, 6.0); ax.set_ylim(0, 19)
    ax.set_xlabel(r'Mean return-temperature reduction $\Delta T_{\mathrm{RT}}$ (K)')
    ax.set_ylabel('Stock-weighted choice share (%), 2030')

    fig.tight_layout()

    # Bubble-size legend, drawn by hand so that each circle and its value label
    # share the same centre line and the vertical gaps stay uniform.
    ref_vals = (5000, 20000, 40000)
    diam = [np.sqrt(80 + v / 1000.0 * 45) for v in ref_vals]   # points
    pos = ax.get_position()
    w_pt = pos.width * fig.get_figwidth() * 72.0
    h_pt = pos.height * fig.get_figheight() * 72.0

    fs, pad, gap, textpad, title_h = 8.0, 8.0, 7.0, 8.0, 13.0
    d_max = max(diam)
    panel_w = 138.0
    panel_h = pad + title_h + sum(diam) + gap * (len(diam) - 1) + pad
    right, top = 0.995, 0.995
    left = right - panel_w / w_pt
    bottom = top - panel_h / h_pt

    ax.add_patch(plt.Rectangle((left, bottom), right - left, top - bottom,
                               transform=ax.transAxes, facecolor='white',
                               edgecolor='#bbbbbb', linewidth=0.6, zorder=10))
    ax.text((left + right) / 2, top - pad / h_pt,
            'Mean investment per building', transform=ax.transAxes,
            ha='center', va='top', fontsize=fs, zorder=12)

    x_c = left + (pad + d_max / 2) / w_pt
    x_lab = left + (pad + d_max + textpad) / w_pt
    y_cur = top - (pad + title_h) / h_pt
    for val, d in zip(ref_vals, diam):
        y_c = y_cur - (d / 2) / h_pt
        ax.scatter([x_c], [y_c], s=80 + val / 1000.0 * 45, c='#e8e8e8',
                   edgecolor='#777777', linewidth=0.6,
                   transform=ax.transAxes, zorder=11, clip_on=False)
        ax.text(x_lab, y_c, f'{val/1000:.0f} kEUR', transform=ax.transAxes,
                ha='left', va='center', fontsize=fs, zorder=12)
        y_cur = y_c - (d / 2 + gap) / h_pt
    save(fig, 'figure4_misalignment')


if __name__ == '__main__':
    make_fig2_baseline_shares()
    make_fig3_stock_adoption()
    make_fig_misalignment()
    make_fig5_cost_decomposition()
    make_fig6_tariff_dose_response()
    make_fig7_renovation_effects()
    make_fig8_sensitivity_panel()
    make_figB1_measure_set_comparison()
    make_figA1_nl_rank()
    print(f"\nAll figures written to {FIG_DIR}")
