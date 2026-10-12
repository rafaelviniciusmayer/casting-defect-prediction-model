"""
Feature engineering for the aluminum die casting dataset.

Reproduces the ~95 engineered columns of
aluminum_diecasting_dataset_with_features.csv from the 15 raw process
variables, returning the 110 features in the order expected by
models/best_model.pkl (artifact['process_vars']).

Range-based features use the defect-free ranges stored in
aluminum_diecasting_dataset_metadata.json (generation_config.variable_ranges).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_PATH = Path(__file__).parent / 'aluminum_diecasting_dataset_metadata.json'

RAW_VARS = [
    'piston_velocity_phase1', 'metal_velocity_gate', 'fill_time',
    'phase_transition_position', 'intensification_time_phase3',
    'intensification_pressure', 'solidification_time', 'cycle_time',
    'sleeve_fill_percentage', 'sleeve_diameter', 'sleeve_length',
    'plunger_lubricant', 'plunger_sleeve_clearance', 'sleeve_temperature',
    'plunger_temperature',
]

EPS = 1e-6


def load_defect_free_ranges(metadata_path=METADATA_PATH):
    """Return {variable: (min_ok, max_ok)} for the continuous variables."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        ranges = json.load(f)['generation_config']['variable_ranges']
    return {
        var: tuple(cfg['defect_free_range'])
        for var, cfg in ranges.items()
        if cfg.get('type') == 'continuous' and 'defect_free_range' in cfg
    }


def load_variable_ranges(metadata_path=METADATA_PATH):
    """Return {variable: (defect_free_range, defect_prone_range)} for the continuous variables."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        ranges = json.load(f)['generation_config']['variable_ranges']
    return {
        var: (tuple(cfg['defect_free_range']), tuple(cfg['defect_prone_range']))
        for var, cfg in ranges.items()
        if cfg.get('type') == 'continuous'
    }


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 110 model features (15 raw + 95 engineered) from raw process variables."""
    missing = [c for c in RAW_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing raw process variables: {missing}")

    x = df[RAW_VARS].astype(float).reset_index(drop=True)
    ranges = load_defect_free_ranges()

    pv1 = x['piston_velocity_phase1']
    vgate = x['metal_velocity_gate']
    fill = x['fill_time']
    t3 = x['intensification_time_phase3']
    press = x['intensification_pressure']
    solid = x['solidification_time']
    cycle = x['cycle_time']
    t_sleeve = x['sleeve_temperature']
    t_plunger = x['plunger_temperature']
    diam = x['sleeve_diameter']
    length = x['sleeve_length']

    feats = {}

    # Ratios and interactions
    feats['velocity_ratio'] = vgate / (pv1 + EPS)
    feats['fill_time_ratio'] = fill / (cycle + EPS)
    feats['solidification_ratio'] = solid / (cycle + EPS)
    feats['pressure_time_ratio'] = press / (t3 + EPS)
    feats['temp_ratio'] = t_sleeve / (t_plunger + EPS)
    feats['sleeve_aspect_ratio'] = length / (diam + EPS)
    feats['intensification_energy'] = press * t3
    feats['velocity_distance'] = pv1 * fill
    feats['temp_solidification'] = (t_sleeve + t_plunger) / 2 * solid
    feats['temp_diff'] = t_sleeve - t_plunger
    feats['cycle_fill_diff'] = cycle - fill
    feats['pressure_deviation'] = press - 100

    # Distance to the defect-free range (normalized by range width)
    for var, (lo, hi) in ranges.items():
        width = hi - lo + EPS
        center = (lo + hi) / 2
        feats[f'{var}_distance_from_center'] = (x[var] - center).abs() / width
        outside = np.maximum(lo - x[var], 0) + np.maximum(x[var] - hi, 0)
        feats[f'{var}_distance_from_range'] = outside / width

    # Range membership flags
    for var, (lo, hi) in ranges.items():
        feats[f'{var}_in_range'] = ((x[var] >= lo) & (x[var] <= hi)).astype(int)
        feats[f'{var}_above_range'] = (x[var] > hi).astype(int)
        feats[f'{var}_below_range'] = (x[var] < lo).astype(int)

    in_range = pd.DataFrame({v: feats[f'{v}_in_range'] for v in ranges})
    center_dist = pd.DataFrame({v: feats[f'{v}_distance_from_center'] for v in ranges})
    feats['n_vars_in_range'] = in_range.sum(axis=1)
    feats['n_vars_out_of_range'] = len(ranges) - feats['n_vars_in_range']
    feats['avg_distance_from_ideal'] = center_dist.mean(axis=1)
    feats['max_distance_from_ideal'] = center_dist.max(axis=1)

    # Physical estimates and transforms
    feats['sleeve_volume_estimate'] = np.pi * (diam / 2) ** 2 * length
    feats['fill_volume_ratio'] = feats['sleeve_volume_estimate'] * x['sleeve_fill_percentage'] / 100
    feats['weighted_velocity'] = 0.3 * pv1 + 0.7 * vgate
    feats['total_process_time'] = fill / 1000 + t3 + solid
    feats['process_efficiency'] = fill / (cycle * 1000 + EPS)
    feats['intensification_pressure_log'] = np.log1p(press)
    feats['cycle_time_log'] = np.log1p(cycle)
    feats['piston_velocity_phase1_squared'] = pv1 ** 2
    feats['metal_velocity_gate_squared'] = vgate ** 2

    out = pd.concat([x, pd.DataFrame(feats)], axis=1)
    out.index = df.index
    return out
