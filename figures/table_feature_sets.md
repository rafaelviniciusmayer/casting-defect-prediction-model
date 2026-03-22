# Feature sets – Feature Engineering

Conjuntos de variáveis utilizadas no pipeline de predição de defeitos (fundição de alumínio). Total de **115 features**, derivadas das **15 variáveis de processo** originais.

---

## Table 3. Feature Engineering Summary: Categories and Counts

| Feature Category | Count | Description |
|------------------|-------|-------------|
| Original variables | 15 | Raw process measurements from sensors. |
| Ratio features | 6 | Proportional relationships (velocity, time, pressure ratios). |
| Product features | 3 | Multiplicative interactions (energy, distance, thermal). |
| Difference features | 2 | Gradients between related variables. |
| Distance features | 30 | Normalized distances from ideal operating ranges. |
| Binary range features | 45 | Categorical position relative to ideal ranges. |
| Statistical aggregation | 4 | Summary measures of overall process state. |
| Domain-specific features | 5 | Casting-specific calculations (volume, efficiency). |
| Mathematical transformations | 5 | Logarithmic and quadratic transformations. |
| **Total features** | **115** | |

---

## 1. Original variables (15 features)

Raw process measurements from sensors.

| # | Feature name | Description |
|---|--------------|-------------|
| 1 | piston_velocity_phase1 | Piston injection velocity during 1st phase (m/s) |
| 2 | metal_velocity_gate | Liquid metal velocity at gate entrance (m/s) |
| 3 | fill_time | Cavity fill time (ms) |
| 4 | phase_transition_position | Phase 1 to Phase 2 transition position (mm) |
| 5 | intensification_time_phase3 | Intensification pressure rise time – Phase 3 (s) |
| 6 | intensification_pressure | Intensification pressure – Phase 3 (MPa) |
| 7 | solidification_time | Solidification time (s) |
| 8 | cycle_time | Total cycle time (s) |
| 9 | sleeve_fill_percentage | Shot sleeve fill percentage (%) |
| 10 | sleeve_diameter | Shot sleeve diameter (mm) |
| 11 | sleeve_length | Shot sleeve length (mm) |
| 12 | plunger_lubricant | Plunger lubrication condition |
| 13 | plunger_sleeve_clearance | Plunger–sleeve clearance (mm) |
| 14 | sleeve_temperature | Shot sleeve temperature (°C) |
| 15 | plunger_temperature | Plunger tip temperature (°C) |

---

## 2. Ratio features (6 features)

Proportional relationships (velocity, time, pressure ratios).

| # | Feature name | Formula |
|---|--------------|--------|
| 1 | velocity_ratio | piston_velocity_phase1 / metal_velocity_gate |
| 2 | fill_time_ratio | fill_time / cycle_time |
| 3 | solidification_ratio | solidification_time / cycle_time |
| 4 | pressure_time_ratio | intensification_pressure × intensification_time_phase3 |
| 5 | temp_ratio | sleeve_temperature / plunger_temperature |
| 6 | sleeve_aspect_ratio | sleeve_length / sleeve_diameter |

---

## 3. Product features (3 features)

Multiplicative interactions (energy, distance, thermal).

| # | Feature name | Formula |
|---|--------------|--------|
| 1 | intensification_energy | intensification_pressure × intensification_time_phase3 |
| 2 | temp_solidification | sleeve_temperature × solidification_time |
| 3 | weighted_velocity | piston_velocity_phase1 × (sleeve_fill_percentage / 100) |

---

## 4. Difference features (2 features)

Gradients between related variables.

| # | Feature name | Formula |
|---|--------------|--------|
| 1 | temp_diff | sleeve_temperature − plunger_temperature |
| 2 | cycle_fill_diff | cycle_time − fill_time |

---

## 5. Distance features (30 features)

Normalized distances from ideal operating ranges.  
15 × distance from center of ideal range + 15 × distance from ideal range boundary.

| # | Feature name | Description |
|---|--------------|-------------|
| 1 | piston_velocity_phase1_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 2 | metal_velocity_gate_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 3 | fill_time_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 4 | phase_transition_position_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 5 | intensification_time_phase3_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 6 | intensification_pressure_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 7 | solidification_time_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 8 | cycle_time_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 9 | sleeve_fill_percentage_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 10 | sleeve_diameter_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 11 | sleeve_length_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 12 | plunger_lubricant_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 13 | plunger_sleeve_clearance_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 14 | sleeve_temperature_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 15 | plunger_temperature_distance_from_center | Absolute distance from value to center of ideal (defect-free) range. |
| 16 | piston_velocity_phase1_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 17 | metal_velocity_gate_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 18 | fill_time_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 19 | phase_transition_position_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 20 | intensification_time_phase3_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 21 | intensification_pressure_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 22 | solidification_time_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 23 | cycle_time_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 24 | sleeve_fill_percentage_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 25 | sleeve_diameter_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 26 | sleeve_length_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 27 | plunger_lubricant_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 28 | plunger_sleeve_clearance_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 29 | sleeve_temperature_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |
| 30 | plunger_temperature_distance_from_range | Distance outside ideal range: 0 if within range; positive if below min or above max. |

---

## 6. Binary range features (45 features)

Categorical position relative to ideal ranges.  
For each of the 15 process variables: in_range, above_range, below_range (0/1).

| # | Feature name | Description |
|---|--------------|-------------|
| 1 | piston_velocity_phase1_in_range | 1 if within defect-free range, else 0. |
| 2 | metal_velocity_gate_in_range | 1 if within defect-free range, else 0. |
| 3 | fill_time_in_range | 1 if within defect-free range, else 0. |
| 4 | phase_transition_position_in_range | 1 if within defect-free range, else 0. |
| 5 | intensification_time_phase3_in_range | 1 if within defect-free range, else 0. |
| 6 | intensification_pressure_in_range | 1 if within defect-free range, else 0. |
| 7 | solidification_time_in_range | 1 if within defect-free range, else 0. |
| 8 | cycle_time_in_range | 1 if within defect-free range, else 0. |
| 9 | sleeve_fill_percentage_in_range | 1 if within defect-free range, else 0. |
| 10 | sleeve_diameter_in_range | 1 if within defect-free range, else 0. |
| 11 | sleeve_length_in_range | 1 if within defect-free range, else 0. |
| 12 | plunger_lubricant_in_range | 1 if within defect-free range, else 0. |
| 13 | plunger_sleeve_clearance_in_range | 1 if within defect-free range, else 0. |
| 14 | sleeve_temperature_in_range | 1 if within defect-free range, else 0. |
| 15 | plunger_temperature_in_range | 1 if within defect-free range, else 0. |
| 16 | piston_velocity_phase1_above_range | 1 if above max of defect-free range, else 0. |
| 17 | metal_velocity_gate_above_range | 1 if above max of defect-free range, else 0. |
| 18 | fill_time_above_range | 1 if above max of defect-free range, else 0. |
| 19 | phase_transition_position_above_range | 1 if above max of defect-free range, else 0. |
| 20 | intensification_time_phase3_above_range | 1 if above max of defect-free range, else 0. |
| 21 | intensification_pressure_above_range | 1 if above max of defect-free range, else 0. |
| 22 | solidification_time_above_range | 1 if above max of defect-free range, else 0. |
| 23 | cycle_time_above_range | 1 if above max of defect-free range, else 0. |
| 24 | sleeve_fill_percentage_above_range | 1 if above max of defect-free range, else 0. |
| 25 | sleeve_diameter_above_range | 1 if above max of defect-free range, else 0. |
| 26 | sleeve_length_above_range | 1 if above max of defect-free range, else 0. |
| 27 | plunger_lubricant_above_range | 1 if above max of defect-free range, else 0. |
| 28 | plunger_sleeve_clearance_above_range | 1 if above max of defect-free range, else 0. |
| 29 | sleeve_temperature_above_range | 1 if above max of defect-free range, else 0. |
| 30 | plunger_temperature_above_range | 1 if above max of defect-free range, else 0. |
| 31 | piston_velocity_phase1_below_range | 1 if below min of defect-free range, else 0. |
| 32 | metal_velocity_gate_below_range | 1 if below min of defect-free range, else 0. |
| 33 | fill_time_below_range | 1 if below min of defect-free range, else 0. |
| 34 | phase_transition_position_below_range | 1 if below min of defect-free range, else 0. |
| 35 | intensification_time_phase3_below_range | 1 if below min of defect-free range, else 0. |
| 36 | intensification_pressure_below_range | 1 if below min of defect-free range, else 0. |
| 37 | solidification_time_below_range | 1 if below min of defect-free range, else 0. |
| 38 | cycle_time_below_range | 1 if below min of defect-free range, else 0. |
| 39 | sleeve_fill_percentage_below_range | 1 if below min of defect-free range, else 0. |
| 40 | sleeve_diameter_below_range | 1 if below min of defect-free range, else 0. |
| 41 | sleeve_length_below_range | 1 if below min of defect-free range, else 0. |
| 42 | plunger_lubricant_below_range | 1 if below min of defect-free range, else 0. |
| 43 | plunger_sleeve_clearance_below_range | 1 if below min of defect-free range, else 0. |
| 44 | sleeve_temperature_below_range | 1 if below min of defect-free range, else 0. |
| 45 | plunger_temperature_below_range | 1 if below min of defect-free range, else 0. |

---

## 7. Statistical aggregation (4 features)

Summary measures of overall process state.

| # | Feature name | Description |
|---|--------------|-------------|
| 1 | n_vars_in_range | Number of process variables within ideal range (0–15) |
| 2 | n_vars_out_of_range | Number of variables out of ideal range |
| 3 | avg_distance_from_ideal | Mean of the 15 distances to ideal center |
| 4 | max_distance_from_ideal | Maximum of the 15 distances to ideal center |

---

## 8. Domain-specific features (5 features)

Casting-specific calculations (volume, efficiency).

| # | Feature name | Description |
|---|--------------|-------------|
| 1 | sleeve_volume_estimate | π × (sleeve_diameter/2)² × sleeve_length |
| 2 | fill_volume_ratio | (sleeve_fill_percentage/100) × (fill_time/cycle_time) |
| 3 | process_efficiency | fill_time / cycle_time |
| 4 | velocity_distance | Sum of distances to ideal center for piston and metal velocities |
| 5 | pressure_deviation | Distance of intensification_pressure from ideal range |

---

## 9. Mathematical transformations (5 features)

Logarithmic and quadratic transformations.

| # | Feature name | Formula |
|---|--------------|--------|
| 1 | intensification_pressure_log | log1p(max(intensification_pressure, 0)) |
| 2 | cycle_time_log | log1p(max(cycle_time, 0)) |
| 3 | piston_velocity_phase1_squared | piston_velocity_phase1² |
| 4 | metal_velocity_gate_squared | metal_velocity_gate² |
| 5 | total_process_time | cycle_time (time-scale feature for model) |
