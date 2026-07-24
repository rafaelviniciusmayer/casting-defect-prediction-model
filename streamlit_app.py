"""
Streamlit Web App - Defect Prediction in Die Casting
====================================================

Interactive web interface for the defect prediction model.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import torch
import json
import pickle
import pandas as pd
from typing import Dict
from train_model import PyTorchModelWrapper, DefectPredictionNN

# 15 original process variables shown in the UI
PROCESS_VARS_15 = [
    'piston_velocity_phase1', 'metal_velocity_gate', 'fill_time',
    'phase_transition_position', 'intensification_time_phase3',
    'intensification_pressure', 'solidification_time', 'cycle_time',
    'sleeve_fill_percentage', 'sleeve_diameter', 'sleeve_length',
    'plunger_lubricant', 'plunger_sleeve_clearance', 'sleeve_temperature',
    'plunger_temperature'
]

# Page configuration
st.set_page_config(
    page_title="Die Casting Defect Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    /* Main theme - Industrial/Metallic feel */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Headers */
    h1 {
        color: #e94560 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: #f1f1f1 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #e94560;
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] div[data-testid="stSlider"] label,
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] label {
        color: #ffffff !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 10px;
        padding: 15px;
    }
    
    [data-testid="stMetric"] label {
        color: #ffffff !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f1f1 !important;
    }
    
    /* Input fields */
    .stSlider > div > div > div {
        color: #e94560;
    }
    
    /* Success/Error boxes */
    .stSuccess {
        background: rgba(0, 200, 83, 0.1);
        border: 1px solid #00c853;
    }
    
    .stError {
        background: rgba(233, 69, 96, 0.1);
        border: 1px solid #e94560;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00c853 0%, #ffc107 50%, #e94560 100%);
    }
    
    /* Custom risk level badges */
    .risk-minimal { background: #00c853; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .risk-low { background: #4caf50; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .risk-moderate { background: #ffc107; color: black; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .risk-high { background: #ff5722; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .risk-critical { background: #e94560; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; animation: pulse 1s infinite; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Variable cards */
    .var-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #e94560;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #ffffff;
        padding: 20px;
        font-size: 12px;
    }
    
    .recommendations-text {
        color: #ffffff !important;
        font-size: 0.95rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model and components (cached)."""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        model = artifacts['model']
        scaler = artifacts['scaler']
        process_vars = artifacts['process_vars']
        defect_cols = artifacts['defect_cols']
        
        # Create compatible metadata
        metadata = {
            # UI uses only the 15 process variables
            'process_variables': PROCESS_VARS_15,
            # Model may expect engineered features (e.g., velocity_ratio)
            'model_variables': process_vars,
            'defects': defect_cols,
            'model_name': artifacts.get('model_name', 'pytorch_model'),
            'metrics': artifacts.get('metrics', {}),
            'prob_stats': artifacts.get('prob_stats', {})
        }
        
        return model, scaler, metadata
        
    except FileNotFoundError:
        st.error("❌ Model not found! Run 'python train_model.py' first.")
        st.stop()


def predict_defects(values: Dict[str, float], model, scaler, metadata) -> Dict:
    """Make defect prediction using the trained PyTorch model."""
    variables = metadata.get('model_variables', metadata['process_variables'])
    defects = metadata['defects']
    
    # Prepare features
    # For engineered variables not present in sidebar inputs, fallback to 0.0
    features = np.array([[values.get(var, 0.0) for var in variables]], dtype=np.float32)
    
    # Run prediction using the model wrapper
    probs = model.predict_proba(features)[0]
    
    # Convert to percentages
    probabilities = {d: round(float(p) * 100, 2) for d, p in zip(defects, probs)}
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    max_risk = max(probabilities.values())
    
    # Risk classification
    if max_risk < 5:
        classification = 'MINIMAL'
    elif max_risk < 15:
        classification = 'LOW'
    elif max_risk < 30:
        classification = 'MODERATE'
    elif max_risk < 50:
        classification = 'HIGH'
    else:
        classification = 'CRITICAL'
    
    return {
        'probabilities': probabilities,
        'sorted_probs': sorted_probs,
        'max_risk': max_risk,
        'classification': classification,
        'probable_defects': [d for d, p in sorted_probs if p >= 10]
    }


def analyze_variables(values: Dict[str, float], metadata) -> Dict:
    """Analyze if variables are within ideal ranges."""
    if 'generation_config' in metadata:
        config = metadata['generation_config'].get('variable_ranges', {})
    else:
        config = metadata.get('variable_config', {})
    
    issues = []
    in_range = []
    
    for var, value in values.items():
        if var not in config:
            continue
        
        var_cfg = config[var]
        
        if var_cfg.get('type') == 'continuous':
            range_ok = var_cfg.get('defect_free_range')
            range_def = var_cfg.get('defect_prone_range')
            
            if range_ok and range_def:
                min_ok, max_ok = range_ok
                min_def, max_def = range_def
                
                if isinstance(min_ok, str):
                    min_ok, max_ok = float(min_ok), float(max_ok)
                    min_def, max_def = float(min_def), float(max_def)
                
                if not (min_ok <= value <= max_ok):
                    if value < min_ok:
                        distance = min_ok - value
                        zone = min_ok - min_def
                        severity = min(distance / zone, 1.0) if zone > 0 else 1.0
                        direction = 'BELOW'
                    else:
                        distance = value - max_ok
                        zone = max_def - max_ok
                        severity = min(distance / zone, 1.0) if zone > 0 else 1.0
                        direction = 'ABOVE'
                    
                    issues.append({
                        'variable': var,
                        'name': var_cfg.get('name', var),
                        'value': value,
                        'unit': var_cfg.get('unit', ''),
                        'ideal_range': (min_ok, max_ok),
                        'severity': round(severity * 100, 1),
                        'direction': direction
                    })
                else:
                    in_range.append({
                        'variable': var,
                        'name': var_cfg.get('name', var),
                        'value': value,
                        'unit': var_cfg.get('unit', ''),
                        'ideal_range': (min_ok, max_ok)
                    })
    
    return {'issues': issues, 'in_range': in_range, 'total_issues': len(issues)}


def get_risk_badge(classification: str) -> str:
    """Return HTML badge for risk classification."""
    badge_class = f"risk-{classification.lower()}"
    return f'<span class="{badge_class}">{classification}</span>'


def main():
    # Header
    st.markdown("# Die Casting Defect Predictor")
    st.markdown("### High-Pressure Die Casting of Aluminum Alloys")
    st.markdown("---")
    
    # Load model
    try:
        model, scaler, metadata = load_model()
        
        # Show model information
        st.sidebar.markdown("### 📊 Model Information")
        st.sidebar.info(f"""
        **Model:** {metadata.get('model_name', 'PyTorch NN').upper()}  
        **F1-Score:** {metadata.get('metrics', {}).get('f1_micro', 0):.4f}  
        **Max Prob.:** {metadata.get('prob_stats', {}).get('max_prob', 0):.4f}
        """)
        
        st.sidebar.success("✨ PyTorch model with defect weighting!")
        
        variables = metadata['process_variables']
        defects = metadata['defects']
        
        # Variable configuration (default values)
        var_config = {
            'piston_velocity_phase1': {
                'name': 'Piston Velocity Phase 1',
                'unit': 'm/s',
                'type': 'continuous',
                'defect_free_range': [0.20, 0.25],
                'defect_prone_range': [0.15, 0.35]
            },
            'metal_velocity_gate': {
                'name': 'Metal Velocity at Gate',
                'unit': 'm/s',
                'type': 'continuous',
                'defect_free_range': [3.5, 3.8],
                'defect_prone_range': [2.5, 5.0]
            },
            'fill_time': {
                'name': 'Fill Time',
                'unit': 'ms',
                'type': 'continuous',
                'defect_free_range': [30, 40],
                'defect_prone_range': [20, 50]
            },
            'phase_transition_position': {
                'name': 'Phase Transition Position',
                'unit': 'mm',
                'type': 'continuous',
                'defect_free_range': [170, 180],
                'defect_prone_range': [160, 190]
            },
            'intensification_time_phase3': {
                'name': 'Intensification Time Phase 3',
                'unit': 's',
                'type': 'continuous',
                'defect_free_range': [6.0, 7.5],
                'defect_prone_range': [5.0, 10.0]
            },
            'intensification_pressure': {
                'name': 'Intensification Pressure',
                'unit': 'MPa',
                'type': 'continuous',
                'defect_free_range': [90, 110],
                'defect_prone_range': [60, 140]
            },
            'solidification_time': {
                'name': 'Solidification Time',
                'unit': 's',
                'type': 'continuous',
                'defect_free_range': [18, 22],
                'defect_prone_range': [10, 35]
            },
            'cycle_time': {
                'name': 'Cycle Time',
                'unit': 's',
                'type': 'continuous',
                'defect_free_range': [70, 80],
                'defect_prone_range': [50, 100]
            },
            'sleeve_fill_percentage': {
                'name': 'Sleeve Fill Percentage',
                'unit': '%',
                'type': 'continuous',
                'defect_free_range': [40, 50],
                'defect_prone_range': [25, 55]
            },
            'sleeve_diameter': {
                'name': 'Sleeve Diameter',
                'unit': 'mm',
                'type': 'continuous',
                'defect_free_range': [46, 49],
                'defect_prone_range': [40, 55]
            },
            'sleeve_length': {
                'name': 'Sleeve Length',
                'unit': 'mm',
                'type': 'continuous',
                'defect_free_range': [120, 130],
                'defect_prone_range': [110, 140]
            },
            'plunger_lubricant': {
                'name': 'Plunger Lubrication',
                'unit': '',
                'type': 'categorical',
                'defect_free_range': [0, 0],
                'defect_prone_range': [0, 2]
            },
            'plunger_sleeve_clearance': {
                'name': 'Plunger-Sleeve Clearance',
                'unit': 'mm',
                'type': 'continuous',
                'defect_free_range': [0.08, 0.12],
                'defect_prone_range': [0.05, 0.20]
            },
            'sleeve_temperature': {
                'name': 'Sleeve Temperature',
                'unit': '°C',
                'type': 'continuous',
                'defect_free_range': [180, 195],
                'defect_prone_range': [150, 220]
            },
            'plunger_temperature': {
                'name': 'Plunger Temperature',
                'unit': '°C',
                'type': 'continuous',
                'defect_free_range': [165, 175],
                'defect_prone_range': [120, 200]
            }
        }
        
        # Add configuration metadata
        metadata['variable_config'] = var_config
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Run 'python train_model.py' first.")
        return
    
    # Sidebar - Process Variables Input
    st.sidebar.markdown("## ⚙️ Process Variables")
    st.sidebar.markdown("Adjust values to simulate process conditions.")
    st.sidebar.markdown("---")
    
    values = {}
    
    # Group variables by stage
    stages = {
        'Filling': ['piston_velocity_phase1', 'metal_velocity_gate', 'fill_time', 'phase_transition_position'],
        'Intensification': ['intensification_time_phase3', 'intensification_pressure'],
        'Solidification': ['solidification_time', 'cycle_time'],
        'Setup': ['sleeve_fill_percentage', 'sleeve_diameter', 'sleeve_length', 'plunger_lubricant', 'plunger_sleeve_clearance'],
        'Temperature': ['sleeve_temperature', 'plunger_temperature']
    }
    
    # Create inputs for each stage
    for stage, stage_vars in stages.items():
        st.sidebar.markdown(f"### 📍 {stage}")
        
        for var in stage_vars:
            if var in variables:  # Check if variable exists in the model
                cfg = var_config.get(var, {})
                name = cfg.get('name', var.replace('_', ' ').title())
                unit = cfg.get('unit', '')
                var_type = cfg.get('type', 'continuous')
                
                if var_type == 'continuous':
                    defect_free = cfg.get('defect_free_range', [0, 100])
                    defect_prone = cfg.get('defect_prone_range', [0, 100])
                    
                    min_val = float(defect_prone[0])
                    max_val = float(defect_prone[1])
                    ideal_min = float(defect_free[0])
                    ideal_max = float(defect_free[1])
                    default_val = (ideal_min + ideal_max) / 2
                    
                    # Determine step based on range
                    range_size = max_val - min_val
                    if range_size < 1:
                        step = 0.01
                    elif range_size < 10:
                        step = 0.1
                    else:
                        step = 1.0
                    
                    values[var] = st.sidebar.slider(
                        f"{name} ({unit})",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step,
                        help=f"Ideal range: {ideal_min} - {ideal_max} {unit}"
                    )
                    
                else:  # categorical
                    options = {
                        0: "Normal (ideal)",
                        1: "Insufficient lubrication",
                        2: "Excessive lubrication"
                    }
                    selected = st.sidebar.selectbox(
                        f"{name}",
                        options=list(options.keys()),
                        format_func=lambda x: options[x],
                        index=0,
                        help="Plunger lubrication condition"
                    )
                    values[var] = selected
        
        st.sidebar.markdown("---")
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📊 Current Process Parameters")
        
        # Show current values in a nice format
        param_df = []
        for var in variables:
            cfg = var_config.get(var, {})
            name = cfg.get('name', var.replace('_', ' ').title())
            unit = cfg.get('unit', '')
            value = values[var]
            
            if cfg.get('type') == 'continuous':
                defect_free = cfg.get('defect_free_range', [0, 100])
                min_ok, max_ok = defect_free
                if min_ok <= value <= max_ok:
                    status = "✅ OK"
                else:
                    status = "⚠️ Out of Range"
            else:
                status = "✅ OK" if value == 0 else "⚠️ Non-ideal"
            
            param_df.append({
                'Parameter': name,
                'Value': f"{value} {unit}",
                'Status': status
            })
        
        df = pd.DataFrame(param_df)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("## 🎯 Quick Stats")
        
        # Count variables inside/outside range
        analysis = analyze_variables(values, metadata)
        in_range = len(analysis['in_range'])
        out_range = analysis['total_issues']
        
        st.metric("Variables In Range", f"{in_range}/{len(variables)}")
        st.metric("Variables Out of Range", f"{out_range}/{len(variables)}")
    
    st.markdown("---")
    
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button(
            "🔍 CHECK DEFECT PROBABILITY",
            type="primary",
            use_container_width=True
        )
    
    if predict_button:
        with st.spinner("Analyzing process parameters..."):
            # DEBUG section - uncomment to see values sent to model
            # st.markdown("---")
            # with st.expander("🔍 DEBUG: Values sent to model", expanded=True):
            #     st.markdown("**Process Variables being sent to predict_defects():**")
            #     debug_df = pd.DataFrame([
            #         {'Variable': var, 'Value': values[var]} 
            #         for var in variables
            #     ])
            #     st.dataframe(debug_df, use_container_width=True, hide_index=True)
            #     st.code(f"values = {values}", language='python')
            
            # Get predictions
            result = predict_defects(values, model, scaler, metadata)
            analysis = analyze_variables(values, metadata)
        
        st.markdown("---")
        st.markdown("### Defect Prediction Dashboard")

        # 2x2 layout in a single view (as requested for Figure 4)
        top_left, top_right = st.columns(2)

        with top_left:
            st.markdown("**Process variables (ideal range)**")
            param_rows = []
            for var in variables:
                if var not in values:
                    continue
                cfg = var_config.get(var, {})
                name = cfg.get('name', var.replace('_', ' ').title())
                unit = cfg.get('unit', '')
                value = values[var]
                if cfg.get('type') == 'continuous':
                    min_ok, max_ok = cfg.get('defect_free_range', [0, 100])
                    ideal = f"{min_ok}-{max_ok} {unit}"
                    status = "✅" if min_ok <= value <= max_ok else "⚠️"
                else:
                    ideal = "-"
                    status = "✅" if value == 0 else "⚠️"
                param_rows.append({
                    "Parameter": name,
                    "Value": f"{value} {unit}",
                    "Ideal": ideal,
                    "Status": status
                })
            st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True, height=210)

        with top_right:
            st.markdown("**Defect probabilities (28 types, threshold)**")
            thresholds = metadata.get('optimal_thresholds', {})
            defect_rows = []
            for defect, prob in result['sorted_probs']:
                th = thresholds.get(defect, 0.5)
                defect_rows.append({
                    "Defect": defect.replace('_', ' ').title(),
                    "P(%)": round(prob, 1),
                    "Thr(%)": round(float(th) * 100, 0)
                })
            st.dataframe(pd.DataFrame(defect_rows), use_container_width=True, hide_index=True, height=210)

        bottom_left, bottom_right = st.columns(2)

        with bottom_left:
            st.markdown("**Outside ideal range (severity)**")
            if analysis['issues']:
                out_rows = []
                for issue in analysis['issues']:
                    out_rows.append({
                        "Variable": issue['name'],
                        "Value": f"{issue['value']} {issue['unit']}",
                        "Ideal": f"{issue['ideal_range'][0]}-{issue['ideal_range'][1]} {issue['unit']}",
                        "Sev%": issue['severity'],
                        "Dir": "Below" if issue['direction'] == 'BELOW' else "Above"
                    })
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True, height=165)
            else:
                st.caption("All variables are within ideal range.")

        with bottom_right:
            st.markdown("**Recommendations**")
            if analysis['issues']:
                recs = []
                for issue in analysis['issues']:
                    direction = "increase" if issue['direction'] == 'BELOW' else "decrease"
                    recs.append(
                        f"- **{issue['name']}**: {direction} toward {issue['ideal_range'][0]}-{issue['ideal_range'][1]} {issue['unit']}"
                    )
                st.markdown('<div class="recommendations-text">' + "<br>".join(recs[:8]) + '</div>', unsafe_allow_html=True)
            else:
                st.caption("No adjustments needed.")
    
    # Footer
    st.markdown("---")
    


if __name__ == "__main__":
    main()
