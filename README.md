# Adoption of return-temperature measures in district heating

Code and data for the paper

> Kök A, Kranzl L, Haas R, Catal J. *Reducing return temperatures in district
> heating: How motivational tariffs and subsidies shape the adoption of
> building-side measures.* Submitted to Applied Energy, 2026.

The model predicts which building-side measures owners of district-heated
multi-family buildings choose, and how motivational return-temperature tariffs,
investment subsidies and building renovation change that choice. It couples
building-stock data from the Invert/EE-Lab model, simulated effects of nine
building-side measures, and a multinomial-logit adoption model, applied to nine
Austrian multi-family archetypes in five renovation states for 2030 and 2040.

The scripts in this repository reproduce every data-driven figure and table
of the paper.

## Quick start

```bash
pip install -r requirements.txt

python run_scenarios.py        # tariff scenarios for both choice sets
python run_paper_analysis.py   # reference case, sensitivities, nested logit
python make_figures.py         # Figures 2-8, A.1, B.1  -> outputs/figures/
python make_tables.py          # Tables 1, 2, 4, 5      -> outputs/tables/
```

The four steps take a few seconds each. They must run in this order,
because each step reads the outputs of the previous ones.
`run_paper_analysis.py` also checks that its reference case agrees with the
no-tariff run of `run_scenarios.py`. Tested with Python 3.13, numpy 2.1,
pandas 2.2, matplotlib 3.10 and openpyxl 3.1.

The result tables are included under `outputs/`, so they can be inspected
without running the model.

## Repository structure

| Path | Content |
|---|---|
| `data/building_measures.xlsx` | Costs, lifetimes and simulated effects of the building-side measures per archetype |
| `data/invert_building_stock_2030.csv`, `..._2040.csv` | Building stock by archetype and renovation state (Invert/EE-Lab) |
| `load_data.py` | Reads the measure workbook |
| `load_invert_data.py` | Reads the building-stock files and maps them to the archetypes |
| `interpolate_measures.py` | Transfers measure effects to renovated buildings (paper Eq. 4) |
| `motivational_tariff.py` | Linear and stepped bonus/malus tariffs (Eq. 7) |
| `analyze_renovations.py` | Total annual heating cost and logit model for one variant (Eqs. 1-3) |
| `logit_model.py` | Multinomial logit with cost-relative utilities (Eq. 3) |
| `nested_logit.py` | Nested logit for the robustness check (Eq. 8) |
| `scenario.py` | Runs all variants and aggregates to archetypes and the stock (Eq. 5) |
| `run_scenarios.py`, `run_paper_analysis.py` | Analysis runs |
| `make_figures.py`, `make_tables.py` | Figures and LaTeX tables of the paper |
| `outputs/scenarios/` | Results per choice set and tariff; tariff sweeps; expected return-temperature reduction |
| `outputs/paper_analysis/` | Reference case, cost decomposition, sensitivities, nested logit, diffusion |
| `outputs/tables/` | Tables of the paper as LaTeX |

## Where each result of the paper comes from

| Paper | Generated file | Underlying results |
|---|---|---|
| Table 1 | `outputs/tables/table_archetypes.tex` | input data |
| Table 2 | `outputs/tables/table_measures.tex` | input data |
| Table 4 | `outputs/tables/table_main_results.tex` | `paper_analysis/base_case/overall_shares_with_diffusion.csv` |
| Table 5 | `outputs/tables/table_nested_logit.tex` | `paper_analysis/nested_logit/nl_robustness.csv` |
| Figure 2 | `figure2_baseline_adoption` | `scenarios/all_measures/per_tariff/none/`, `paper_analysis/base_case/` |
| Figure 3 | `figure3_stock_adoption` | `paper_analysis/base_case/overall_shares_with_diffusion.csv` |
| Figure 4 | `figure4_misalignment` | `scenarios/all_measures/per_tariff/none/`, input data |
| Figure 5 | `figure5_cost_decomposition` | `paper_analysis/base_case/cost_decomposition_2030.csv` |
| Figure 6 | `figure6_tariff_dose_response` | `scenarios/all_measures/sweep_summary/` |
| Figure 7 | `figure7_renovation_effects` | `scenarios/all_measures/per_tariff/none/results_renovation_analysis_*.csv` |
| Figure 8 | `figure8_sensitivity` | `paper_analysis/sensitivity/`, `paper_analysis/m6_sensitivity/` |
| Figure A.1 | `figureA1_nl_rank` | `paper_analysis/nested_logit/nl_robustness.csv` |
| Figure B.1 | `figureB1_measure_set_comparison` | `scenarios/restricted/per_tariff/none/` |
| Expected return-temperature reduction (Section 5.3) | | `scenarios/all_measures/expected_rt_reduction.csv` |
| Subsidy results (Section 5.4) | | `paper_analysis/sensitivity/lambda_price_sensitivity.csv` |

Table 3 (economic parameters) and Figure 1 (framework diagram) are not
generated from data.

## Data

### Building-side measures (`data/building_measures.xlsx`)

The effects of the measures were quantified in coupled simulations of the
buildings and their heating systems: dynamic multi-zone building models
(IDA ICE) combined with hydraulic models of the heating system and the
district-heating substation (Dymola/Modelica). The simulation study is
described in

> Catal J, Müller A, Heimrath R, Kranzl L, Dahash A, Kalfa M, Mihaly N,
> Schmidt R-R. Optimized buildings for decarbonized district heating: A
> measures catalogue for reducing temperatures, enhancing flexibility, and
> cutting costs. Book of Abstracts, 11th International Conference on Smart
> Energy Systems, 2025.

The workbook keeps its original German sheet names. It contains only the
sheets the model reads:

| Sheet | Quantity |
|---|---|
| `Kosten_1-Investitionen` | Investment cost (EUR per building) |
| `Kosten_2-laufende Kosten` | Operating cost (EUR per year) |
| `Kosten_3-Lebensdauern` | Lifetime (years) |
| `Effekte_1a-VT-Red` | Supply-temperature reduction (K); read but not used in the paper |
| `Effekte_2a-RT-Red` | Return-temperature reduction (K) |
| `Effekte_2b-RT` | Return temperatures; the annual-mean baseline per archetype parameterizes the tariffs |
| `Effekte_3-Effizienz` | Efficiency improvement (% of final energy demand before the measure) |

The sheets list a wider catalogue of measures. The model uses these nine:

| Paper | Workbook ID | Measure |
|---|---|---|
| M1 | 1.3 | Flow-limiter adjustment |
| M2 | 1.5 | Heat-exchanger renovation |
| M3 | 2.1 | Hydraulic balancing |
| M4 | 2.3 | Pump downsizing |
| M5 | 2.5 | Buffer storage |
| M6 | 2.7 | User-behavior improvement |
| M7 | 2.8 | Pump downsizing + flow limiter |
| M8 | 2.9 | Pump downsizing + flow limiter + hydraulic balancing |
| M9 | 2.10 | Hydraulic balancing + heat exchanger |

The header cells of the two cost sheets give the unit per m² of heated floor
area. The model uses the investment values as costs per building (for
example 975-2,625 EUR for the flow limiter, Table 2 of the paper). The
operating costs are zero for all nine measures.

Not all values are simulation results. Buffer-storage investment costs are
literature values set in `load_data.py` (`FALLBACK_INVESTMENT_COSTS`), and the
costs and the flat 9.6 % efficiency of the user-behavior measure are
assumptions. The paper discusses both (Section 4.2, Table 2).

### Building stock (`data/invert_building_stock_<year>.csv`)

One row per archetype and renovation state, from the Invert/EE-Lab
building-stock model (Müller 2015; Kranzl et al. 2013, Energy Policy 59:44-58).
The model uses these columns:

| Column | Meaning |
|---|---|
| `name` | Invert building category (BCAT_5/6/7 = 6/12/24 dwellings) and construction period |
| `BC_last_action` | Renovation state |
| `number_of_buildings` | Number of represented buildings, used as aggregation weight |
| `hwb_norm` | Specific space-heating demand after renovation (kWh m⁻² a⁻¹), used to interpolate measure effects |
| `fed_sh_per_bssh` | Final energy demand (kWh a⁻¹), used for the energy cost |

The remaining columns are further Invert outputs that the model does not use.
The paper orders the renovation states by depth, which is not the order of the
Invert labels:

| Invert label | Paper |
|---|---|
| `no action` | No action |
| `maintenance` | Maintenance |
| `renovation 2` | Moderate renovation |
| `renovation stand` | Standard renovation |
| `renovation 1` | Deep renovation |

## License

MIT, see [LICENSE](LICENSE).

## Funding

This work was supported by the Austrian Research Promotion Agency (FFG)
[grant number FO999894612] within the project "Risk minimization for
decarbonizing heating networks via network temperature reductions and
flexibility utilization" (DeRiskDH).
