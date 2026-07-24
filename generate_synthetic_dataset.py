"""
Synthetic Dataset Generator for Defect Prediction in Aluminum Die Casting
==========================================================================

VERSION 1.0

This script generates a synthetic dataset based on expert knowledge mapping
from Process Engineers specializing in High-Pressure Die Casting (HPDC)
of aluminum alloys.

DATA SOURCE:
- Expert mapping from die casting process engineers
- Operating ranges (defect-free and defect-prone) defined empirically
- Influence matrix (0-3) based on domain expertise
- Defect probability percentages validated by engineers

STATISTICAL APPROACH:
- Values generated using uniform distribution within defect-free ranges (for normal scenarios)
- Uniform distribution reflects that all values within acceptable range are equally valid
- No assumption of central preference - more conservative and realistic for controlled processes
- Variable-defect correlation follows the influence matrix

ASSUMPTIONS (Documented):
- Values WITHIN defect-free range: very low defect probability
- Values IN defect zone: probability increases proportionally
- Severity is proportional to distance from ideal value
- Multiple out-of-range variables have combined effect (not simple addition)
- Influence weights (0→0%, 1→33%, 2→67%, 3→100%) are linear interpolations
  of the ordinal scale provided by experts
- HIGH INFLUENCE (2-3): Any deviation from ideal has minimum probability
  (3% for influence 2, 5% for influence 3) - reflects expert knowledge

DISTRIBUTION:
- 94% good samples: values within or near defect-free range
- 6% defect scenarios: values outside ideal ranges

STRICT LIMITS:
- Values NEVER exceed min/max of defect-prone range
- No outliers beyond expert-defined limits

OUTPUT PHILOSOPHY:
- Dataset simulates real factory data collection
- Each row = one produced part ("snapshot")
- Only observable data: process variables + defect outcomes
- No derived columns (_zone, _severity, prob_*) - model learns these patterns

Author: Rafael Vinicius Mayer
Date: December 2025
Version: 1.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# EXPERT MAPPING DATA
# Source: Defects and Influences in High-Pressure Die Casting of Aluminum Alloys
# =============================================================================

PROCESS_VARIABLES = {
    'piston_velocity_phase1': {
        'name': 'Piston injection velocity during 1st phase',
        'stage': 'Injection Phase 1',
        'sensor': 'Position/Velocity Sensor (PLC)',
        'unit': 'm/s',
        'defect_free_range': (0.20, 0.25),
        'defect_prone_range': (0.15, 0.30),
        'defect_chance_pct': (10, 25),
        'type': 'continuous'
    },
    'metal_velocity_gate': {
        'name': 'Liquid metal velocity at gate entrance',
        'stage': 'Injection Phase 1',
        'sensor': 'Position/Flow Velocity Sensor (PLC)',
        'unit': 'm/s',
        'defect_free_range': (3.5, 3.8),
        'defect_prone_range': (2.8, 4.5),
        'defect_chance_pct': (12, 28),
        'type': 'continuous'
    },
    'fill_time': {
        'name': 'Cavity fill time',
        'stage': 'Injection Phase 1',
        'sensor': 'Timer/Chronometer (PLC)',
        'unit': 'ms',
        'defect_free_range': (30, 40),
        'defect_prone_range': (20, 50),
        'defect_chance_pct': (8, 22),
        'type': 'continuous'
    },
    'phase_transition_position': {
        'name': 'Phase 1 to Phase 2 transition position',
        'stage': 'Phase Transition 1-2',
        'sensor': 'Piston Position Sensor (PLC)',
        'unit': 'mm',
        'defect_free_range': (170, 180),
        'defect_prone_range': (160, 190),
        'defect_chance_pct': (5, 15),
        'type': 'continuous'
    },
    'intensification_time_phase3': {
        'name': 'Intensification pressure rise time (Phase 3)',
        'stage': 'Intensification Phase 3',
        'sensor': 'Timer/Chronometer (PLC)',
        'unit': 's',
        'defect_free_range': (5, 8),
        'defect_prone_range': (3, 10),
        'defect_chance_pct': (4, 12),
        'type': 'continuous'
    },
    'intensification_pressure': {
        'name': 'Intensification pressure (Phase 3)',
        'stage': 'Intensification Phase 3',
        'sensor': 'Pressure Sensor/Manometer (PLC)',
        'unit': 'MPa',
        'defect_free_range': (80, 120),
        'defect_prone_range': (60, 140),
        'defect_chance_pct': (6, 18),
        'type': 'continuous'
    },
    'solidification_time': {
        'name': 'Solidification time',
        'stage': 'Cooling',
        'sensor': 'Timer/Chronometer (PLC)',
        'unit': 's',
        'defect_free_range': (15, 25),
        'defect_prone_range': (10, 35),
        'defect_chance_pct': (3, 10),
        'type': 'continuous'
    },
    'cycle_time': {
        'name': 'Total cycle time',
        'stage': 'Complete Cycle',
        'sensor': 'Timer/Chronometer (PLC)',
        'unit': 's',
        'defect_free_range': (60, 90),
        'defect_prone_range': (55, 100),
        'defect_chance_pct': (2, 8),
        'type': 'continuous'
    },
    'sleeve_fill_percentage': {
        'name': 'Shot sleeve fill percentage',
        'stage': 'Setup/Maintenance',
        'sensor': 'Level Meter/Visual Inspection (Calculated)',
        'unit': '%',
        'defect_free_range': (40, 50),
        'defect_prone_range': (30, 60),
        'defect_chance_pct': (3, 10),
        'type': 'continuous'
    },
    'sleeve_diameter': {
        'name': 'Shot sleeve diameter',
        'stage': 'Setup/Maintenance',
        'sensor': 'Caliper/Micrometer',
        'unit': 'mm',
        'defect_free_range': (45, 50),
        'defect_prone_range': (42, 52),
        'defect_chance_pct': (2, 8),
        'type': 'continuous'
    },
    'sleeve_length': {
        'name': 'Shot sleeve length',
        'stage': 'Setup/Maintenance',
        'sensor': 'Caliper/Steel Ruler',
        'unit': 'mm',
        'defect_free_range': (120, 130),
        'defect_prone_range': (115, 135),
        'defect_chance_pct': (2, 7),
        'type': 'continuous'
    },
    'plunger_lubricant': {
        'name': 'Plunger lubrication condition',
        'stage': 'Setup/Maintenance',
        'sensor': 'Timer (PLC) and Visual Inspection',
        'unit': 'category',
        # 0 = Adequate (60% circumference coverage, no excess in chamber)
        # 1 = Insufficient lubrication
        # 2 = Excess lubrication in chamber
        'defect_free_values': [0],
        'defect_prone_values': [1, 2],
        'defect_chance_pct': (8, 20),
        'type': 'categorical'
    },
    'plunger_sleeve_clearance': {
        'name': 'Plunger-sleeve clearance',
        'stage': 'Setup/Maintenance',
        'sensor': 'Caliper/Micrometer/Ruler',
        'unit': 'mm',
        'defect_free_range': (0.08, 0.12),
        'defect_prone_range': (0.05, 0.18),
        'defect_chance_pct': (15, 35),
        'type': 'continuous'
    },
    'sleeve_temperature': {
        'name': 'Shot sleeve temperature',
        'stage': 'Setup/Maintenance',
        'sensor': 'Thermocouple/Pyrometer/Thermal Camera',
        'unit': '°C',
        'defect_free_range': (180, 200),
        'defect_prone_range': (150, 220),
        'defect_chance_pct': (7, 18),
        'type': 'continuous'
    },
    'plunger_temperature': {
        'name': 'Plunger tip temperature',
        'stage': 'Setup/Maintenance',
        'sensor': 'Thermocouple/Pyrometer/Thermal Camera',
        'unit': '°C',
        'defect_free_range': (160, 180),
        'defect_prone_range': (120, 200),
        'defect_chance_pct': (6, 16),
        'type': 'continuous'
    }
}

# =============================================================================
# DEFECTS - As mapped by process engineers
# =============================================================================

DEFECTS = [
    'blisters_post_treatment',      # Blisters after treatment
    'surface_blisters',             # Blisters
    'die_sticking',                 # Die sticking
    'flow_lines',                   # Flow lines
    'surface_streaks',              # Surface streaks
    'cold_shut',                    # Cold shut
    'heat_cracks',                  # Heat cracks
    'ejector_pin_marks',            # Ejector pin marks
    'die_soldering',                # Die soldering
    'surface_oxide_inclusions',     # Surface oxide inclusions
    'low_tensile_strength',         # Low tensile strength
    'low_elongation',               # Low elongation
    'low_ultimate_strength',        # Low ultimate strength
    'low_fatigue_resistance',       # Low fatigue resistance
    'low_surface_hardness',         # Low surface hardness
    'density_deviation',            # Inadequate density
    'incomplete_fill',              # Incomplete fill
    'flash',                        # Flash
    'warpage',                      # Warpage
    'shrinkage_porosity',           # Shrinkage porosity
    'volumetric_variation',         # Volumetric variations
    'dimensional_deviation',        # Dimensional deviations
    'gas_porosity',                 # Gas porosity
    'gas_bubbles',                  # Gas bubbles
    'internal_shrinkage',           # Internal shrinkage
    'cracks',                       # Cracks
    'hard_inclusions',              # Hard inclusions
    'oxide_inclusions'              # Oxide inclusions
]

# =============================================================================
# INFLUENCE MATRIX - EXACTLY as mapped by experts
# Values: 0 = None/Unknown, 1 = Low/Possible, 2 = Strong, 3 = Very Strong
# =============================================================================

VARIABLE_ORDER = list(PROCESS_VARIABLES.keys())

# Matrix extracted directly from expert mapping table
# Rows: Process variables, Columns: Defects (in DEFECTS order)
INFLUENCE_MATRIX = {
    'piston_velocity_phase1':       [1,1,0,0,0,2,0,0,0,0,0,0,0,0,0,2,2,0,0,0,0,0,2,2,0,0,0,1],
    'metal_velocity_gate':          [2,2,2,1,0,2,0,0,1,2,2,2,2,2,2,2,2,2,0,0,1,1,2,2,0,0,0,2],
    'fill_time':                    [0,0,0,2,1,3,0,0,0,0,2,2,2,2,2,2,3,0,0,0,0,0,2,0,0,0,0,0],
    'phase_transition_position':    [0,0,1,2,0,2,0,0,0,0,2,2,2,2,2,2,2,0,0,0,0,0,2,2,0,0,0,0],
    'intensification_time_phase3':  [2,2,0,0,0,0,0,0,1,0,1,1,1,1,0,2,0,1,1,0,2,2,2,2,2,0,0,0],
    'intensification_pressure':     [3,1,2,0,0,0,0,0,2,0,1,1,1,1,0,2,0,1,1,0,3,3,3,2,2,0,0,0],
    'solidification_time':          [0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,1,0,0,2,0,2,2,1,0,0,2,0,0],
    'cycle_time':                   [0,1,1,1,0,2,1,0,2,0,1,1,1,0,1,0,1,1,1,2,2,2,1,0,0,2,0,0],
    'sleeve_fill_percentage':       [2,2,0,0,0,1,0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0],
    'sleeve_diameter':              [1,1,0,0,0,2,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,2,0,0,0,0],
    'sleeve_length':                [1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,0,2,2,0,0,0,1],
    'plunger_lubricant':            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,2,0,0,2,2],
    'plunger_sleeve_clearance':     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    'sleeve_temperature':           [0,0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,1,0],
    'plunger_temperature':          [0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0]
}

# =============================================================================
# INFLUENCE WEIGHT MAPPING
# This converts the ordinal scale (0-3) to probability multipliers
# DOCUMENTED ASSUMPTION: Linear interpolation of expert scale
# =============================================================================

INFLUENCE_WEIGHTS = {
    0: 0.00,  # No influence
    1: 0.33,  # Low/Possible influence (1/3 of max effect)
    2: 0.67,  # Strong influence (2/3 of max effect)
    3: 1.00   # Very strong influence (full effect)
}
# Note: Linear interpolation of ordinal scale (0-3) to continuous weights.
# This is the simplest and least assumptive transformation.


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def generate_continuous_value(var_config: Dict, 
                               defect_scenario: bool = False,
                               seed: Optional[int] = None) -> Tuple[float, str, float]:
    """
    Generates a value for a continuous process variable.
    
    Distribution strategy (adjusted for 94% good / 6% defect ratio):
    - If defect_scenario=False (~94%): values uniformly distributed within defect-free range
      Uses uniform distribution to reflect that all values within acceptable range are
      equally valid in controlled industrial processes (no central preference assumed)
    - If defect_scenario=True (~6%): values in transition/problem zone
    
    IMPORTANT: Values are STRICTLY limited to [min_def, max_def] range.
    No values can exceed the expert-defined defect-prone limits.
    
    Args:
        var_config: Variable configuration dictionary
        defect_scenario: Whether this sample should potentially have defects
        seed: Random seed for reproducibility
        
    Returns:
        Tuple (value, zone, severity)
        - zone: 'normal', 'transition', 'problem'
        - severity: 0.0 to 1.0 (how problematic the value is)
    """
    if seed is not None:
        np.random.seed(seed)
    
    min_ok, max_ok = var_config['defect_free_range']
    min_def, max_def = var_config['defect_prone_range']
    
    if not defect_scenario:
        # GOOD SCENARIO (94%): ALL values within defect-free range
        # This ensures ~0% defect probability for these samples
        # Using uniform distribution: all values within acceptable range are equally valid
        # This reflects real industrial processes where operators may use different
        # setpoints within the acceptable range, and there's no physical reason
        # to prefer central values over boundary values
        value = np.random.uniform(min_ok, max_ok)
        zone = 'normal'
    else:
        # DEFECT SCENARIO (6%): Values outside defect-free range
        dice = np.random.random()
        
        if dice < 0.4:
            # MILD: Just outside ideal range (30-60% severity)
            if np.random.random() < 0.5:
                offset = np.random.uniform(0.3, 0.6) * (min_ok - min_def)
                value = min_ok - offset
            else:
                offset = np.random.uniform(0.3, 0.6) * (max_def - max_ok)
                value = max_ok + offset
            zone = 'transition'
            
        elif dice < 0.8:
            # MODERATE: Significantly outside (60-90% severity)
            if np.random.random() < 0.5:
                offset = np.random.uniform(0.6, 0.9) * (min_ok - min_def)
                value = min_ok - offset
            else:
                offset = np.random.uniform(0.6, 0.9) * (max_def - max_ok)
                value = max_ok + offset
            zone = 'transition'
            
        else:
            # SEVERE: Near or at limits (90-100% severity)
            if np.random.random() < 0.5:
                offset = np.random.uniform(0.9, 1.0) * (min_ok - min_def)
                value = min_ok - offset
            else:
                offset = np.random.uniform(0.9, 1.0) * (max_def - max_ok)
                value = max_ok + offset
            zone = 'problem'
    
    # STRICT LIMIT: Never exceed defect-prone range
    value = np.clip(value, min_def, max_def)
    
    severity = calculate_severity(value, var_config)
    return value, zone, severity


def generate_categorical_value(var_config: Dict,
                                defect_scenario: bool = False) -> Tuple[int, str, float]:
    """
    Generates a value for a categorical process variable.
    
    Args:
        var_config: Variable configuration dictionary
        defect_scenario: Whether this sample should potentially have defects
        
    Returns:
        Tuple (value, zone, severity)
    """
    if defect_scenario and np.random.random() < 0.5:
        # In defect scenario, 50% chance of problematic value
        value = np.random.choice(var_config['defect_prone_values'])
        zone = 'problem'
        severity = 0.8
    else:
        value = np.random.choice(var_config['defect_free_values'])
        zone = 'normal'
        severity = 0.0
    
    return value, zone, severity


def calculate_severity(value: float, var_config: Dict) -> float:
    """
    Calculates severity (0-1) based on distance from defect-free range.
    
    Severity is a normalized measure of how "out of ideal" a value is.
    - 0.0 = within defect-free range (ideal)
    - 0.0-0.5 = in transition zone (mild to moderate)
    - 0.5-1.0 = near or beyond limits (severe)
    
    Returns:
        Severity between 0.0 and 1.0 (capped at 1.0, never exceeds)
    """
    if var_config['type'] == 'categorical':
        if value in var_config['defect_free_values']:
            return 0.0
        return 0.8
    
    min_ok, max_ok = var_config['defect_free_range']
    min_def, max_def = var_config['defect_prone_range']
    
    # Within defect-free range = no problem
    if min_ok <= value <= max_ok:
        return 0.0
    
    # Calculate normalized distance
    if value < min_ok:
        transition_zone = min_ok - min_def
        distance = min_ok - value
        if transition_zone > 0:
            severity = distance / transition_zone
        else:
            severity = 1.0
    else:
        transition_zone = max_def - max_ok
        distance = value - max_ok
        if transition_zone > 0:
            severity = distance / transition_zone
        else:
            severity = 1.0
    
    # Cap severity at 1.0 (no values exceed defect-prone range)
    return min(severity, 1.0)


def calculate_defect_probability(severity: float, 
                                  influence: int, 
                                  chance_pct: Tuple[float, float] = None) -> float:
    """
    Calculates probability of a specific defect occurring.
    
    Based ONLY on the expert influence matrix:
    - Severity: how far the value is from the good range (0.0 = in good range, 1.0 = worst case)
    - Influence: influence level of this variable on this specific defect (0, 1, 2, 3)
    
    CORRECT INTERPRETATION:
    - When severity=1.0 (worst case) and influence=3 (very strong influence): probability ≈ 100%
    - When severity=1.0 and influence=2 (strong influence): probability ≈ 75%
    - When severity=1.0 and influence=1 (low influence): probability ≈ 40%
    - When influence=0: this variable does NOT affect this defect = 0%
    - When severity=0: variable is in the good range = 0%
    
    Formula:
    - If influence = 0: P = 0% (variable does not affect this defect)
    - If severity = 0: P = 0% (variable is in the good range)
    - Otherwise: P = severity × max_prob_by_influence[influence]
    
    Example (piston_velocity_phase1):
    - Worst case (severity=1.0) with influence 3: P = 1.0 × 0.95 = 95% (close to 100%)
    - Worst case (severity=1.0) with influence 2: P = 1.0 × 0.75 = 75%
    - Worst case (severity=1.0) with influence 1: P = 1.0 × 0.40 = 40%
    - Worst case (severity=1.0) with influence 0: P = 0% (no effect)
    
    Args:
        severity: Distance from ideal range (0.0 = in good range, 1.0 = worst case at limits)
        influence: Influence level on this specific defect (0, 1, 2, 3)
        chance_pct: Parameter kept for compatibility, but not used in the calculation
    
    Returns:
        Probability between 0.0 and 0.95
    """
    # Influence 0 = this variable does NOT affect this specific defect
    if influence == 0:
        return 0.0
    
    # Severity 0 = variable is in the good range (no defect)
    if severity == 0:
        return 0.0
    
    # Maximum probability based ONLY on influence (when severity=1.0)
    # Based on the expert influence matrix
    max_prob_by_influence = {
        1: 0.40,   # Low influence: up to 40% in worst case
        2: 0.75,   # Strong influence: up to 75% in worst case
        3: 0.95    # Very strong influence: up to 95% in worst case (close to 100%)
    }
    
    max_prob = max_prob_by_influence[influence]
    
    # Direct formula: linear scale with severity up to max_prob
    # When severity=1.0 and influence=3: P = 0.95 (close to 100%)
    probability = severity * max_prob
    
    # For high influences (2-3), ensure minimum probability even with low severity
    # This reflects that high-influence variables are very sensitive
    if influence >= 2:
        if influence == 2:
            # Strong influence: minimum 50% of maximum possible even with low severity
            min_effective_prob = 0.50 * max_prob
        else:  # influence == 3
            # Very strong influence: minimum 60% of maximum possible even with low severity
            min_effective_prob = 0.60 * max_prob
        
        # Ensure effective minimum probability
        probability = max(min_effective_prob, probability)
    
    # Cap at 95% to leave margin for combination with other variables
    return min(probability, 0.95)


def combine_probabilities(probabilities: List[float]) -> float:
    """
    Combines multiple independent probabilities.
    
    Uses probability union formula:
    P(A ∪ B) = P(A) + P(B) - P(A) * P(B)
    
    This approach prevents combined probability from exceeding 1.0
    and correctly models non-mutually exclusive events.
    """
    total = 0.0
    for p in probabilities:
        if p > 0:
            total = total + p - (total * p)
    return min(total, 1.0)


# =============================================================================
# SAMPLE GENERATION
# =============================================================================

def select_variables_for_defect() -> List[str]:
    """
    Selects how many variables will fall outside the ideal range.
    
    Distribution based on expert knowledge:
    Events with 3+ variables out of range are very rare and usually cause
    process interruption (serious machine, mold, or process problems).
    
    Target distribution:
    - 65%: only 1 variable out (majority of cases)
    - 27%: 2 variables out (moderate cases)
    - 5%: 3 variables out (rare cases)
    - 3%: 4+ variables out (very rare events, usually cause interruption)
    
    Returns:
        List of variable names selected to fall outside the range
    """
    all_variables = list(PROCESS_VARIABLES.keys())
    dice = np.random.random()
    
    if dice < 0.65:
        # 65%: only 1 variable out
        n_vars = 1
    elif dice < 0.92:  # 65% + 27% = 92%
        # 27%: 2 variables out
        n_vars = 2
    elif dice < 0.97:  # 92% + 5% = 97%
        # 5%: 3 variables out (rare cases)
        n_vars = 3
    else:
        # 3%: 4+ variables out (very rare events)
        # Distribution within this group: 4 (40%), 5 (30%), 6 (20%), 7 (10%)
        n_vars = np.random.choice([4, 5, 6, 7], p=[0.4, 0.3, 0.2, 0.1])
    
    # Randomly select n_vars variables
    selected = np.random.choice(all_variables, size=min(n_vars, len(all_variables)), replace=False)
    return selected.tolist()


def generate_sample(defect_scenario: bool = False) -> Dict:
    """
    Generates a complete dataset sample.
    
    Args:
        defect_scenario: If True, generate values likely to cause defects.
                        If False, generate values within ideal ranges.
        
    Returns:
        Dictionary with all values and probabilities
    """
    sample = {}
    severities = {}
    
    # If defect scenario, determine which variables will be out of control
    if defect_scenario:
        vars_out_of_control = select_variables_for_defect()
    else:
        vars_out_of_control = []
    
    # Generate values for each process variable
    for var_name, var_config in PROCESS_VARIABLES.items():
        # Only put this variable out of control if it was selected
        var_defect_scenario = var_name in vars_out_of_control
        
        if var_config['type'] == 'continuous':
            value, zone, severity = generate_continuous_value(var_config, var_defect_scenario)
        else:
            value, zone, severity = generate_categorical_value(var_config, var_defect_scenario)
        
        sample[var_name] = value
        sample[f'{var_name}_zone'] = zone
        sample[f'{var_name}_severity'] = severity
        severities[var_name] = severity
    
    # Calculate probability for each defect
    for idx, defect in enumerate(DEFECTS):
        probs_per_variable = []
        
        for var_name in VARIABLE_ORDER:
            influence = INFLUENCE_MATRIX[var_name][idx]
            severity = severities[var_name]
            
            # Calculation based ONLY on severity and influence (from expert matrix)
            prob = calculate_defect_probability(severity, influence)
            if prob > 0:
                probs_per_variable.append(prob)
        
        # Combine probabilities from all variables
        total_prob = combine_probabilities(probs_per_variable)
        
        # CRITICAL: If no variable has influence on this defect, probability must be 0
        # This ensures defects only occur when at least one variable with influence is out of range
        if len(probs_per_variable) == 0:
            total_prob = 0.0
        
        # Determine if defect occurred (binomial event)
        # Note: We don't save prob_{defect} - it's only used to generate the binary label
        sample[defect] = int(np.random.random() < total_prob)
    
    # Aggregate metrics
    sample['total_defects'] = sum(sample[d] for d in DEFECTS)
    sample['has_defect'] = int(sample['total_defects'] > 0)
    sample['variables_out_of_range'] = sum(
        1 for v in PROCESS_VARIABLES.keys() 
        if sample[f'{v}_zone'] != 'normal'
    )
    
    return sample


def generate_dataset(n_samples: int = 25000,
                      defect_ratio: float = 0.06,
                      seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates the complete synthetic dataset.
    
    Distribution strategy:
    - ~94% of samples: good production (values within ideal ranges)
    - ~6% of samples: defect scenarios (values outside ideal ranges)
    
    Args:
        n_samples: Number of samples to generate (default: 25,000)
        defect_ratio: Ratio of samples with defect scenarios (default: 6%)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (production_df, analysis_df):
        - production_df: Clean dataset simulating real factory data collection
        - analysis_df: Full dataset with internal calculations for analysis
    """
    np.random.seed(seed)
    
    n_defect_samples = int(n_samples * defect_ratio)
    n_good_samples = n_samples - n_defect_samples
    
    print(f"Generating {n_samples:,} samples...")
    print(f"  - Good samples (94%): {n_good_samples:,}")
    print(f"  - Defect scenario samples (6%): {n_defect_samples:,}")
    
    samples = []
    
    # Generate good samples first
    for i in range(n_good_samples):
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i + 1:,}/{n_samples:,} (good samples)")
        
        sample = generate_sample(defect_scenario=False)
        sample['id'] = i + 1
        samples.append(sample)
    
    # Generate defect scenario samples
    for i in range(n_defect_samples):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {n_good_samples + i + 1:,}/{n_samples:,} (defect samples)")
        
        sample = generate_sample(defect_scenario=True)
        sample['id'] = n_good_samples + i + 1
        samples.append(sample)
    
    # Shuffle to mix good and defect samples
    np.random.shuffle(samples)
    
    # Reassign IDs after shuffle
    for i, sample in enumerate(samples):
        sample['id'] = i + 1
    
    df_full = pd.DataFrame(samples)
    
    # Full DataFrame for internal analysis (includes _zone, _severity, prob_*)
    cols_analysis = ['id'] + list(PROCESS_VARIABLES.keys())
    cols_analysis += [f'{v}_zone' for v in PROCESS_VARIABLES.keys()]
    cols_analysis += [f'{v}_severity' for v in PROCESS_VARIABLES.keys()]
    cols_analysis += DEFECTS
    cols_analysis += ['total_defects', 'has_defect', 'variables_out_of_range']
    df_analysis = df_full[[c for c in cols_analysis if c in df_full.columns]]
    
    # Clean DataFrame simulating real factory data:
    # - Process variable values (sensor readings)
    # - Defect outcomes (quality inspection results)
    # 
    # Note: _zone, _severity, and prob_* columns are INTERNAL calculation helpers
    # and are NOT exported - they wouldn't exist in real production data.
    # Each row represents a "snapshot" of a produced part.
    cols_production = ['id'] + list(PROCESS_VARIABLES.keys()) + DEFECTS + ['total_defects', 'has_defect']
    df_production = df_full[[c for c in cols_production if c in df_full.columns]]
    
    return df_production, df_analysis


def analyze_dataset(df_analysis: pd.DataFrame) -> None:
    """
    Statistical analysis of generated dataset.
    
    Uses the internal analysis DataFrame (with _zone, _severity columns)
    to provide insights about the generation process.
    """
    print("\n" + "="*70)
    print("DATASET STATISTICAL ANALYSIS")
    print("="*70)
    
    print(f"\nTotal samples: {len(df_analysis):,}")
    
    # Distribution by zone (internal metric - not in production dataset)
    print(f"\n--- INTERNAL: ZONE DISTRIBUTION (sample of 5 variables) ---")
    for var in list(PROCESS_VARIABLES.keys())[:5]:
        zone_col = f'{var}_zone'
        if zone_col in df_analysis.columns:
            counts = df_analysis[zone_col].value_counts()
            print(f"\n  {var}:")
            for zone, count in counts.items():
                print(f"    {zone}: {count:,} ({100*count/len(df_analysis):.1f}%)")
    
    # Range verification
    print(f"\n--- RANGE VERIFICATION ---")
    for var, config in PROCESS_VARIABLES.items():
        if config['type'] == 'continuous':
            min_val = df_analysis[var].min()
            max_val = df_analysis[var].max()
            min_def, max_def = config['defect_prone_range']
            
            margin = (max_def - min_def) * 0.1
            within = (min_val >= min_def - margin) and (max_val <= max_def + margin)
            
            status = "[OK]" if within else "[WARNING] extreme outliers"
            print(f"  {var}: [{min_val:.4f}, {max_val:.4f}] {status}")
    
    # Defect distribution
    print(f"\n--- DEFECT DISTRIBUTION ---")
    print(f"With defects: {df_analysis['has_defect'].sum():,} ({100*df_analysis['has_defect'].mean():.1f}%)")
    print(f"Without defects: {(~df_analysis['has_defect'].astype(bool)).sum():,} ({100*(1-df_analysis['has_defect'].mean()):.1f}%)")
    
    # Distribution of variables out of range (should follow: 65% with 1, 27% with 2, 5% with 3, 3% with 4+)
    print(f"\n--- DISTRIBUTION OF VARIABLES OUT OF RANGE (Defect Scenarios Only) ---")
    defect_scenarios = df_analysis[df_analysis['variables_out_of_range'] > 0]
    if len(defect_scenarios) > 0:
        total_defect_scenarios = len(defect_scenarios)
        for n_out in range(1, 8):
            subset = defect_scenarios[defect_scenarios['variables_out_of_range'] == n_out]
            if len(subset) > 0:
                pct = 100 * len(subset) / total_defect_scenarios
                if n_out == 1:
                    expected = "65%"
                elif n_out == 2:
                    expected = "27%"
                elif n_out == 3:
                    expected = "5%"
                elif n_out >= 4:
                    if n_out == 4:
                        expected = "1.2% (40% of 3%)"
                    elif n_out == 5:
                        expected = "0.9% (30% of 3%)"
                    elif n_out == 6:
                        expected = "0.6% (20% of 3%)"
                    elif n_out == 7:
                        expected = "0.3% (10% of 3%)"
                    else:
                        expected = "part of 3%"
                else:
                    expected = ""
                print(f"  {n_out} variable{'s' if n_out > 1 else ''} out: {pct:.1f}% ({len(subset):,} samples) [expected: ~{expected}]")
    
    # Correlation: variables out of range vs defects (internal metric)
    print(f"\n--- INTERNAL: VARIABLES OUT OF RANGE vs DEFECTS ---")
    for n_out in range(6):
        subset = df_analysis[df_analysis['variables_out_of_range'] == n_out]
        if len(subset) > 0:
            pct_defect = 100 * subset['has_defect'].mean()
            print(f"  {n_out} variables out: {pct_defect:.1f}% with defects ({len(subset):,} samples)")
    
    # Top defects
    print(f"\n--- TOP 10 DEFECTS ---")
    defect_freq = df_analysis[DEFECTS].sum().sort_values(ascending=False).head(10)
    for defect, count in defect_freq.items():
        print(f"  {defect}: {count:,} ({100*count/len(df_analysis):.2f}%)")


def save_dataset(df_production: pd.DataFrame, 
                 df_analysis: pd.DataFrame,
                 base_name: str = 'aluminum_diecasting_dataset') -> None:
    """
    Saves the dataset in multiple formats.
    
    The production dataset simulates real factory data collection:
    - Each row = one produced part ("snapshot")
    - Columns = process variables + defect outcomes
    - No derived/calculated columns (_zone, _severity, prob_*)
    """
    # Round numeric columns to 2 decimal places and format with decimal point
    df_rounded = df_production.copy()
    numeric_cols = df_rounded.select_dtypes(include=[np.number]).columns
    # Exclude integer columns (id, binary flags) from rounding
    exclude_cols = ['id', 'total_defects', 'has_defect'] + DEFECTS
    numeric_cols_to_round = [col for col in numeric_cols if col not in exclude_cols]
    
    # Round to 2 decimal places
    df_rounded[numeric_cols_to_round] = df_rounded[numeric_cols_to_round].round(2)
    
    # Format numbers to always show at least one decimal place (e.g., 159.0 instead of 159)
    # This ensures consistent decimal point formatting in CSV with 2 decimal places max
    def format_number(x):
        if pd.isna(x):
            return x
        # Convert to float
        x_float = float(x)
        # Check if it's effectively an integer (within rounding tolerance)
        if abs(x_float - round(x_float)) < 0.001:
            # Format as integer with .0
            return f'{round(x_float):.1f}'
        else:
            # Format with up to 2 decimal places, removing trailing zeros
            formatted = f'{x_float:.2f}'.rstrip('0')
            # If ends with decimal point, add 0
            if formatted.endswith('.'):
                formatted = formatted + '0'
            return formatted
    
    # Apply formatting to numeric columns (convert to string for CSV)
    for col in numeric_cols_to_round:
        df_rounded[col] = df_rounded[col].apply(format_number)
    
    # Main CSV - Clean production data (what would be collected in a factory)
    # Use decimal='.' to ensure point as decimal separator
    df_rounded.to_csv(f'{base_name}.csv', index=False, decimal='.')
    print(f"\n[OK] Saved: {base_name}.csv")
    print(f"  - {len(df_rounded.columns)} columns: id + {len(PROCESS_VARIABLES)} process vars + {len(DEFECTS)} defects + summary")
    
    # NumPy arrays for PyTorch (use original precision for training)
    features = df_production[list(PROCESS_VARIABLES.keys())].values
    labels = df_production[DEFECTS].values
    
    np.save(f'{base_name}_features.npy', features)
    np.save(f'{base_name}_labels.npy', labels)
    print(f"[OK] Saved: {base_name}_features.npy ({features.shape})")
    print(f"[OK] Saved: {base_name}_labels.npy ({labels.shape})")
    
    # Complete metadata (includes internal config for reproducibility)
    metadata = {
        'version': '1.0',
        'description': 'Synthetic dataset for defect prediction in aluminum die casting',
        'source': 'Expert mapping from die casting process engineers',
        'dataset_philosophy': 'Each row represents a produced part snapshot - only observable data',
        'columns': {
            'process_variables': list(PROCESS_VARIABLES.keys()),
            'defects': DEFECTS,
            'summary': ['total_defects', 'has_defect']
        },
        'generation_config': {
            'variable_ranges': {k: {kk: vv for kk, vv in v.items() 
                                   if kk not in ['defect_free_values', 'defect_prone_values']} 
                              for k, v in PROCESS_VARIABLES.items()},
            'influence_matrix': INFLUENCE_MATRIX,
            'influence_weights': INFLUENCE_WEIGHTS
        },
        'n_samples': len(df_production),
        'statistics': {
            'defect_rate': float(df_production['has_defect'].mean()),
            'avg_defects_per_sample': float(df_production['total_defects'].mean())
        }
    }
    
    with open(f'{base_name}_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] Saved: {base_name}_metadata.json")


def main():
    """
    Main function.
    """
    print("="*70)
    print("SYNTHETIC DATASET GENERATOR v1.0")
    print("Defect Prediction in High-Pressure Die Casting of Aluminum Alloys")
    print("="*70)
    print("\n[INFO] Based on expert mapping from process engineers")
    print("[INFO] Distribution: 94% good samples / 6% defect scenarios")
    print("[INFO] Strict limits: values never exceed min/max defect-prone range")
    print("[INFO] Output: Clean production data (no derived columns)")
    
    # Parameters
    N_SAMPLES = 25000
    DEFECT_RATIO = 0.06  # 6% defect scenarios, 94% good
    
    print(f"\n[PARAMETERS]")
    print(f"   - Total samples: {N_SAMPLES:,}")
    print(f"   - Good samples: {int(N_SAMPLES * (1-DEFECT_RATIO)):,} ({(1-DEFECT_RATIO)*100:.0f}%)")
    print(f"   - Defect scenarios: {int(N_SAMPLES * DEFECT_RATIO):,} ({DEFECT_RATIO*100:.0f}%)")
    
    # Generate (returns production data + internal analysis data)
    df_production, df_analysis = generate_dataset(
        n_samples=N_SAMPLES, 
        defect_ratio=DEFECT_RATIO, 
        seed=42
    )
    
    # Analyze (uses internal data with zone/severity info)
    analyze_dataset(df_analysis)
    
    # Save (exports clean production data)
    save_dataset(df_production, df_analysis)
    
    print("\n" + "="*70)
    print("[SUCCESS] DATASET GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\n[INFO] Dataset structure (simulating real factory data):")
    print(f"   - {len(PROCESS_VARIABLES)} process variables (sensor readings)")
    print(f"   - {len(DEFECTS)} defect columns (quality inspection results)")
    print(f"   - Each row = 1 produced part snapshot")
    
    return df_production


if __name__ == "__main__":
    main()

