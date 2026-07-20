"""TYNDP 2026 -> Tulipa pipeline — central configuration.

Single source of truth for the binding choices from the Decision Register.
Weather year is ONE value here (never hardcoded downstream), per the Register.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------- repo layout
REPO = Path(__file__).resolve().parent
# Point this straight at your data folder (edit if you move it).
# Default is the Windows OneDrive location; override on another machine by setting
# the TYNDP_DATA_2026 environment variable (does not change this default path).
DATA_2026 = Path(os.environ.get(
    "TYNDP_DATA_2026",
    Path.home() / "Nextcloud" / "ExperimentData" / "DATA_2026",))

# Sub-trees (the doubled folder names are how the archive unpacked)
PEMMDB_DIR   = DATA_2026 / "PEMMDB_2.X" / "PEMMDB_2.X"
DEMAND_DIR   = DATA_2026 / "Demand" / "Demand"
PECD_DIR     = DATA_2026 / "PECD" / "PECD"
LINE_DIR     = DATA_2026 / "Line-data" / "Line-data"
NODES_XLSX   = DATA_2026 / "Nodes" / "Nodes" / "LIST OF NODES.xlsx"
HYDROGEN_DIR = DATA_2026 / "Hydrogen" / "Hydrogen"
HEAT_DIR     = DATA_2026 / "Heat" / "Heat"
COMMON_DATA  = DATA_2026 / "PEMMDB_CommonData.xlsx"
CO2_FACTORS  = DATA_2026 / "CO2_emission_factors_in_TYNDP2026.xlsx"
FUEL_PRICES  = DATA_2026 / "TYNDP2026Scenarios_Fuel_CO2_Prices.xlsx"
# per-node fixed/flexible split ('EV_flex_share' %). In this 2026 bundle it ships under
# Electric-Vehicle/Electric-Vehicle/ (not Demand/); fall back to the Demand/ location if present.
EV_FLEX_SHARE = DATA_2026 / "Electric-Vehicle" / "Electric-Vehicle" / "EV_FLEX_SHARE.xlsx"
if not EV_FLEX_SHARE.exists() and (DEMAND_DIR / "EV_FLEX_SHARE.xlsx").exists():
    EV_FLEX_SHARE = DEMAND_DIR / "EV_FLEX_SHARE.xlsx"

TEMPLATE_JSON = REPO / "input-schemas2.json"
OUTPUT_DIR    = REPO / "tulipa_input_north_sea_2026"
INTERMEDIATE_DIR = REPO / "intermediate_2026"

# ---------------------------------------------------------------- binding decisions
SCENARIO = "NationalTrends"     # only scenario present in the 2026 data set
SCENARIO_PRICE_KEY = "NT"       # sheet/column key used in price & CO2 workbooks
SELECTED_YEAR = 2040            # first-build horizon (Register: all four; build 2040 first)

# ---------------------------------------------------------------- perimeter (switchable)
# PERIMETER selects the modelled footprint. "north_sea" reproduces the original 7-node build
# byte-for-byte; "ring" is the pan-European footprint = every node with FULL data coverage
# (electricity demand + PEMMDB generation + Line-data grid links + node-list entry), derived from
# the Step-1 coverage matrix (intermediate_2026/coverage_matrix_ALL.csv). The ONLY place the
# footprint is defined — every reader scopes off COUNTRY_PREFIXES / ONSHORE_NODES via in_scope().
PERIMETER = "ring"   # "north_sea" | "ring"

# Nodes EXCLUDED from "ring" despite a PEMMDB file, because they fail a coverage column (Step 1):
#   FR15, LUF1, LUV1            -> no electricity-demand sheet / not in node list (gen-only artifacts)
#   AZ,DZ,EG,GE,IL,LY,MA,PS,SA,TN -> external (no demand, not in node list): stay as the boundary edge.
_RING_ONSHORE = [
    "AL00", "AT00", "BA00", "BE00", "BG00", "CH00", "CY00", "CZ00", "DE00", "DKE1", "DKW1",
    "EE00", "ES00", "FI00", "FR00", "GR00", "GR03", "HR00", "HU00", "IE00",
    "ITCA", "ITCN", "ITCS", "ITN1", "ITS1", "ITSA", "ITSI", "LT00", "LUG1", "LV00",
    "MD00", "ME00", "MK00", "MT00", "NL00", "NOM1", "NON1", "NOS1", "NOS2", "NOS3",
    "PL00", "PT00", "RO00", "RS00", "SE01", "SE02", "SE03", "SE04", "SI00", "SK00",
    "TR00", "UA00", "UK00", "UKNI",
]

_PERIMETERS = {
    "north_sea": dict(
        country_prefixes=["NL", "BE", "DE", "UK", "DK"],
        onshore_nodes=["NL00", "BE00", "DE00", "UK00", "UKNI", "DKE1", "DKW1"],
        # out-of-perimeter neighbours imported as priced slabs (mutually exclusive with the full set)
        boundary_nodes={"FR": ["FR00"], "NO": ["NOS2", "NOSF"], "SE": ["SE03", "SE04"]},
        boundary_price={"FR": 50.0, "NO": 30.0, "SE": 35.0},
    ),
    "ring": dict(
        country_prefixes=sorted({n[:2] for n in _RING_ONSHORE}),
        onshore_nodes=_RING_ONSHORE,
        # Almost all of interconnected Europe is now internal. The only remaining external
        # neighbours are the North-African / Middle-Eastern / Caucasus nodes (DZ, MA, TN, LY, EG,
        # IL, PS, SA, AZ, GE), which ship NO demand sheet and are NOT in the node list -> they are
        # left as un-modelled external borders (their import edges are dropped, NOT fabricated with
        # a made-up price). Boundary set is EMPTY so it can never overlap the full perimeter.
        boundary_nodes={},
        boundary_price={},
    ),
}
_P = _PERIMETERS[PERIMETER]
# Country prefixes used by common.in_scope() (filters demand, H2, SMR, storage, H2 cross-border)
COUNTRY_PREFIXES = _P["country_prefixes"]
# Onshore electricity market nodes in scope (offshore folded radially into these)
ONSHORE_NODES = _P["onshore_nodes"]

# ---------------------------------------------------------------- weather year (single, first build)
# The demand / PECD sheets carry many weather scenarios as columns "WS###".
# FIRST BUILD: a single average/central year. If exactly one WS column is
# populated we use it; otherwise we pick the central one. Never hardcoded past here.
WEATHER_MODE = "single_central"   # later: "three_weighted"
# The ONE canonical weather column, pinned explicitly (2026-07 input audit). TYNDP NT+ 2040 runs
# exactly three climate years — WS065=CMR5_2039, WS071=ECE3_2035, WS077=ECE3_2041 (the only
# populated columns in the demand files, and the three the KPI dashboard reports). Demand/heat/
# thermal files are populated ONLY for those three, so the old central-populated heuristic picked
# WS071 there — but PECD VRE and hydro files are populated for ALL 30 columns, where the same
# heuristic silently picked WS076 (ECE3_2040, a 10-22% windier year in NW-EU): RES/hydro weather
# did not match demand weather. Pinning WS071 here aligns every input family on one climate year;
# files where WS071 is absent/empty fall back to the central populated column (flagged).
CANONICAL_WS_LABEL = "WS071"      # SSP245_ECE3_2035 — swap to WS065/WS077 for the other TYNDP CYs
WEATHER_TAG = ("single weather year - not probability-weighted, "
               "not for adequacy conclusions")

# ---------------------------------------------------------------- physics constants
H2_LHV_MWH_PER_KG = 0.03333       # 1 kg H2 = 33.33 kWh (LHV)
GJ_PER_MWH = 3.6                  # 1 MWh = 3.6 GJ  (heat-rate / fuel-price unit bridge)

# CO2 price seed comes from the per-scenario price workbook (TYNDP value), not here.

# ---------------------------------------------------------------- run controls
ENABLE_INVESTMENT = False         # Register: pure dispatch run
ENABLE_HEAT_HHP = True            # Register: include HHPs (two heat nodes/country)
ENABLE_DSR = True                 # Register: DSR as priced producer
ENABLE_EV_AS_LOAD = True          # Register/gap: EV folded into exogenous demand
# The EV_FIXED files hold ONLY the fixed (imposed) charging share, split by LOCATION
# (street -> MARKET file, home -> PROSUMER file). The flexible/optimised share is endogenous
# in PLEXOS and exists in NO time-series file. Until Tulipa models demand response, we add the
# flexible share back as ADDITIONAL FIXED load, grossing the fixed series up PER NODE by
# flex/(1-flex), where the flex fraction is read from EV_FLEX_SHARE.xlsx (never hardcoded; for
# NL/DE/BE 2040 it is 50% -> factor 1.0 -> EV energy doubles, matching the ETM passenger total).
# Flip EV_FLEX_AS_FIXED=False once DSR / flexible-EV exists, and route that energy to a shiftable
# asset instead of folding it into fixed demand (otherwise the flexible half is silently dropped).
EV_FLEX_AS_FIXED = True
EV_FLEX_TAG = ("flexible passenger-EV charging added back as FIXED load, per-node gross-up "
               "flex/(1-flex) read from EV_FLEX_SHARE — interim until Tulipa models demand "
               "response; reuses the fixed profile shape (overstates peak coincidence; "
               "conservative on storage/peaker need).")
# Storage efficiencies (Task H — avoid lossless wash trades). Per-flow (charge & discharge);
# round-trip = product. PHS ~0.875x0.875=0.766 RTE; battery defaults if PEMMDB has none.
PHS_FLOW_EFFICIENCY = 0.875       # pumped-hydro charge & discharge (matches OBZ reference)
BATTERY_DEFAULT_RTE = 0.90        # used only if PEMMDB battery efficiency is missing
ENABLE_H2_IMPORT_BACKSTOP = True  # H2 load-shed producer priced at the H2 import (commodity) price
H2_IMPORT_MARKUP = 1.0            # multiply the H2 commodity price (1.0 = import price as-is)
# TYNDP two-band extra-EU H2 imports (H2 IMPORTS GENERATORS PROPERTIES/PROFILES): LTC band at
# 0 EUR/MWh (take-or-pay, capped by contract/profile) + flexible band at corridor marginal cost.
# When ON, the flat 71.9 backstop above is DISABLED (it would underprice the marginal bands);
# feasibility stays guaranteed by the universal ENS at VoLL.
ENABLE_H2_IMPORT_BANDS = True
# Offshore wind hubs (PEMMDB *_OFF nodes): how to represent them.
#   "radial" — fold each hub's wind onto its host onshore zone; add the hub's external cables to
#              that zone's interconnectors (simple; pools capacity at the onshore node).
#   "hub"    — model each hub as its own zero-demand node with its wind + real cables to neighbours
#              (faithful topology; enforces each hub's own cable limits).
OFFSHORE_HUB_MODE = "radial"
# Hub mode = offshore bidding zones (each cabled hub its own zero-demand zone with real cables).
# It writes to a SEPARATE folder so the radial dataset is never overwritten — the two model
# variants coexist and either can be fed to Tulipa.
if OFFSHORE_HUB_MODE == "hub":
    OUTPUT_DIR = REPO / "tulipa_input_north_sea_2026_obz"
# Boundary electricity imports from out-of-perimeter neighbours (sourced from the active perimeter).
ENABLE_BOUNDARY_IMPORTS = True
BOUNDARY_NEIGHBOR_NODES = _P["boundary_nodes"]   # external node(s) -> country tag (empty for "ring")
BOUNDARY_IMPORT_PRICE = _P["boundary_price"]     # EUR/MWh representative import price (tunable knob)

# Guardrail #3 (mutual exclusion): a country must never be both a full perimeter node AND a priced
# boundary import — that double-counts it. Assert the boundary tags don't intersect the full set.
_boundary_full_overlap = {cc for cc in BOUNDARY_NEIGHBOR_NODES
                          if any(n[:2] == cc for n in ONSHORE_NODES)}
assert not _boundary_full_overlap, (
    f"PERIMETER '{PERIMETER}': boundary neighbours {_boundary_full_overlap} are also modelled as "
    "full nodes — remove them from BOUNDARY_NEIGHBOR_NODES/BOUNDARY_IMPORT_PRICE (no double-count).")

# Unit-commitment core (PEMMDB Thermal: Number of units / Min stable power / Ramp rates).
# OFF -> exactly the current LP build (thermal = one aggregated unit, no UC fields).
# ON  -> per-unit capacity (total/units) + initial_units = unit count, unit_commitment method,
#        min_operating_point (= min stable %/100) and ramping limits (MW/h -> p.u./h of unit size).
# Assets missing min-stable/ramps in PEMMDB keep those fields empty (schema defaults; flagged) —
# no fabricated values. Min up/down times & start-up costs exist in PEMMDB_CommonData but have NO
# Tulipa construct in this schema -> NOT included (documented gap).
# Endogenous synthetic fuels (TYNDP NT+ replication): 2 EU demand nodes (sng / e-liquids, flat,
# MWh H2-equivalent) + per-country synthesis conversions drawing from the SHARED {cc}h2 nodes
# (corridor caps from SYNTHETIC FUEL LINES 2026.xlsx) + unconstrained imports at the TYNDP offer
# prices. CO2/BECCS handled as a post-check (signed off), not explicit flows. OFF -> baseline.
ENABLE_SYNFUELS = True

ENABLE_UNIT_COMMITMENT = True
UC_METHOD = "basic"               # schema oneOf: 'none' | 'basic' | '3var'
UC_INTEGER_UNITS = False          # False = relaxed (LP) units_on — recommended first; True = MILP

# Universal Energy-Not-Served (ENS) backstop — mirrors the 2024 build.
# One 'ens' producer wired by transport flow to EVERY electricity & hydrogen demand
# node, so the balance can always close; unmet demand shows up as ens usage in results.
ENABLE_ENS = True
ENS_CAPACITY_MW = 1_000_000.0     # huge -> never the binding limit (2024 value)
ENS_PRICE_EUR_PER_MWH = 1000.0    # VoLL penalty: above any real generator (2024 value)

# Approximations / deviations accumulated at runtime for the final audit
INHERITED_ASSUMPTIONS: list[str] = []
APPROXIMATIONS: list[str] = []

def note_assumption(msg: str) -> None:
    if msg not in INHERITED_ASSUMPTIONS:
        INHERITED_ASSUMPTIONS.append(msg)

def note_approximation(msg: str) -> None:
    if msg not in APPROXIMATIONS:
        APPROXIMATIONS.append(msg)
