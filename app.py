"""
Transportation Problem Solver - Streamlit Web App
VAM + MODI Method with Step-by-Step Visualization
Hospital Medical Supplies Distribution Example

Author: barira zaib
Date: 2025-12-05
"""

import streamlit as st
import pandas as pd
import numpy as np
from transport_solver import TransportModel
import time

# Page config
st.set_page_config(
    page_title="Hospital Transportation Problem Solver",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    div[data-testid="stDataFrame"] {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vam_step' not in st.session_state:
    st.session_state.vam_step = 0
if 'modi_started' not in st.session_state:
    st.session_state.modi_started = False
if 'problem_data' not in st.session_state:
    st.session_state.problem_data = None

def initialize_model(cost_matrix, supply, demand):
    """Initialize the transport model"""
    try:
        model = TransportModel(cost_matrix, supply, demand)
        st.session_state.model = model
        st.session_state.vam_step = 0
        st.session_state.modi_started = False
        st.session_state.problem_data = {
            'cost': cost_matrix,
            'supply': supply,
            'demand': demand
        }
        return True
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return False

def display_cost_matrix(model):
    """Display cost matrix with current supply/demand"""
    st.markdown('<p class="sub-header">📊 Shipping Cost Matrix (per unit)</p>', unsafe_allow_html=True)
    
    cost_df = pd.DataFrame(
        model.cost,
        columns=[f'Hospital {i+1}' for i in range(model.n)],
        index=[f'Warehouse {i+1}' for i in range(model.m)]
    )
    st.dataframe(cost_df.style.highlight_max(axis=None, color='#ffcccb').format("{:.2f}"), 
                 use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Current Supply (units):** {[round(x, 2) for x in model.supply]}")
    with col2:
        st.info(f"**Current Demand (units):** {[round(x, 2) for x in model.demand]}")
    
    if model.dummy_added:
        st.warning(f"⚠️ Dummy {'warehouse' if model.dummy_added == 'row' else 'hospital'} added to balance the problem")

def display_allocation_matrix(model):
    """Display current allocation matrix"""
    st.markdown('<p class="sub-header">📦 Allocation Matrix (Medical Supplies)</p>', unsafe_allow_html=True)
    
    alloc_data = []
    for i in range(model.m):
        row = []
        for j in range(model.n):
            val = model.alloc[i][j]
            if model.is_basic[i][j]:
                row.append(f"{val:.2f}*")
            else:
                row.append(f"{val:.2f}")
        alloc_data.append(row)
    
    alloc_df = pd.DataFrame(
        alloc_data,
        columns=[f'Hospital {i+1}' for i in range(model.n)],
        index=[f'Warehouse {i+1}' for i in range(model.m)]
    )
    st.dataframe(alloc_df, use_container_width=True)
    st.caption("* indicates basic variables (active shipping routes)")
    
    total_cost = model.total_cost()
    st.markdown(f"""
        <div class="metric-card">
            <h2>💰 Total Transportation Cost</h2>
            <h1>${total_cost:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)

def display_vam_step(step_data):
    """Display VAM step information"""
    if step_data is None:
        return
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"**✅ Step {st.session_state.vam_step}:** Allocated **{step_data['alloc_qty']:.2f} units** "
                f"from **Warehouse {step_data['chosen_cell'][0]+1}** to **Hospital {step_data['chosen_cell'][1]+1}** "
                f"with cost **${step_data['cost_cell']:.2f}/unit**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Warehouse Penalties:**")
        pen_row = {f"W{i+1}": round(p, 2) for i, p in enumerate(step_data['penalty_row']) if p >= 0}
        st.json(pen_row)
    with col2:
        st.write("**Hospital Penalties:**")
        pen_col = {f"H{j+1}": round(p, 2) for j, p in enumerate(step_data['penalty_col']) if p >= 0}
        st.json(pen_col)

def display_uv_method(model):
    """Display UV method results"""
    st.markdown('<p class="sub-header">🔍 MODI/UV Method</p>', unsafe_allow_html=True)
    
    u, v = model.compute_uv()
    red = model.reduced_costs()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**U Potentials (Warehouses):**")
        u_dict = {f"u{i+1}": round(val, 4) if val is not None else 'N/A' for i, val in enumerate(u)}
        st.json(u_dict)
    with col2:
        st.write("**V Potentials (Hospitals):**")
        v_dict = {f"v{j+1}": round(val, 4) if val is not None else 'N/A' for j, val in enumerate(v)}
        st.json(v_dict)
    
    st.write("**Reduced Costs Matrix:**")
    red_data = []
    for i in range(model.m):
        row = []
        for j in range(model.n):
            if red[i][j] is not None:
                row.append(round(red[i][j], 4))
            else:
                row.append('N/A')
        red_data.append(row)
    
    red_df = pd.DataFrame(
        red_data,
        columns=[f'Hospital {i+1}' for i in range(model.n)],
        index=[f'Warehouse {i+1}' for i in range(model.m)]
    )
    st.dataframe(red_df, use_container_width=True)
    
    # Check for entering variable
    pos, val, _ = model.find_entering()
    if pos:
        st.markdown(f'<div class="warning-box">⚡ **Entering Cell:** Warehouse {pos[0]+1} → Hospital {pos[1]+1} with reduced cost {val:.4f}</div>', 
                    unsafe_allow_html=True)
        return True
    else:
        st.markdown('<div class="success-box">✅ **Optimal Solution Reached!** All reduced costs are non-negative.</div>', 
                    unsafe_allow_html=True)
        return False

# Main App
st.markdown('<p class="main-header">🏥 Hospital Medical Supplies Transportation Solver</p>', unsafe_allow_html=True)
st.markdown("**VAM (Vogel's Approximation Method) + MODI (Modified Distribution Method)**")
st.markdown("*Optimizing medical supplies delivery from warehouses to hospitals*")
st.divider()

# Sidebar for input
with st.sidebar:
    st.header("📝 Problem Setup")
    
    # Dimensions
    m = st.number_input("Number of Warehouses", min_value=2, max_value=10, value=10)
    n = st.number_input("Number of Hospitals", min_value=2, max_value=10, value=10)
    
    st.divider()
    
    # Load example button
    if st.button("🏥 Load Hospital Example (10×10)", use_container_width=True):
        st.session_state.example_loaded = True
        st.rerun()
    
    st.divider()
    
    # About section
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool solves the **Hospital Medical Supplies Transportation Problem**:
        
        **Problem Context:**
        - 10 warehouses supplying medical equipment
        - 10 hospitals requiring supplies
        - Minimize total shipping cost
        
        **Solution Method:**
        
        1. **VAM (Initial Solution)**
           - Calculates row/column penalties
           - Allocates to minimum cost cells
           - Provides feasible starting solution
        
        2. **MODI/UV Method (Optimization)**
           - Computes dual variables (u, v)
           - Finds entering variables
           - Optimizes until convergence
        
        **Created by:** Shahem Riaz  
        **Date:** December 2025
        """)

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Input Data", "🔄 VAM Solution", "⚡ MODI Optimization"])

with tab1:
    st.markdown('<p class="sub-header">Enter Problem Data</p>', unsafe_allow_html=True)
    
    # Load example data if requested
    if 'example_loaded' in st.session_state and st.session_state.example_loaded:
        # Hospital transportation problem: 10 warehouses × 10 hospitals
        m, n = 10, 10
        
        # Supply from 10 warehouses (medical supplies in units)
        example_supply = [100, 90, 80, 110, 95, 85, 105, 88, 92, 98]
        
        # Demand at 10 hospitals (medical supplies in units)
        example_demand = [80, 70, 60, 90, 85, 75, 95, 82, 88, 78]
        
        # Cost matrix (shipping cost per unit from each warehouse to each hospital)
        example_cost = [
            [12, 15, 18, 10, 14, 16, 13, 11, 17, 19],
            [14, 11, 16, 12, 15, 18, 14, 13, 12, 16],
            [16, 13, 14, 15, 11, 12, 16, 17, 14, 15],
            [10, 14, 17, 11, 13, 15, 12, 14, 16, 18],
            [15, 12, 13, 14, 10, 14, 15, 16, 13, 14],
            [17, 16, 15, 13, 14, 11, 17, 15, 14, 13],
            [13, 15, 12, 16, 17, 14, 10, 12, 15, 16],
            [11, 14, 16, 12, 15, 13, 14, 9, 11, 17],
            [14, 13, 15, 14, 12, 16, 13, 14, 10, 15],
            [18, 17, 14, 15, 16, 15, 18, 16, 14, 11]
        ]
        
        st.session_state.example_loaded = False
        st.success("✅ Hospital example loaded! 10 warehouses × 10 hospitals")
        st.info(f"📊 Total Supply: {sum(example_supply)} units | Total Demand: {sum(example_demand)} units")
    else:
        example_cost = [[0]*n for _ in range(m)]
        example_supply = [0]*m
        example_demand = [0]*n
    
    # Cost matrix input
    st.write("**Shipping Cost Matrix ($/unit)** - Warehouses × Hospitals")
    cost_matrix = []
    for i in range(m):
        cols = st.columns(n)
        row = []
        for j in range(n):
            with cols[j]:
                val = st.number_input(f"W{i+1}→H{j+1}", value=float(example_cost[i][j]), 
                                     key=f"cost_{i}_{j}", label_visibility="collapsed")
                row.append(val)
        cost_matrix.append(row)
    
    st.divider()
    
    # Supply and demand
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Supply (units from each warehouse)**")
        supply = []
        for i in range(m):
            val = st.number_input(f"Warehouse {i+1} Supply", value=float(example_supply[i]), 
                                 key=f"supply_{i}")
            supply.append(val)
    
    with col2:
        st.write("**Demand (units needed at each hospital)**")
        demand = []
        for j in range(n):
            val = st.number_input(f"Hospital {j+1} Demand", value=float(example_demand[j]), 
                                 key=f"demand_{j}")
            demand.append(val)
    
    st.divider()
    
    # Validate and create model
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Build Model", use_container_width=True, type="primary"):
            total_supply = sum(supply)
            total_demand = sum(demand)
            
            if abs(total_supply - total_demand) > 100:
                st.warning(f"⚠️ Unbalanced problem! Supply={total_supply:.2f}, Demand={total_demand:.2f}")
                st.info("The model will automatically add a dummy warehouse/hospital to balance.")
            
            if initialize_model(cost_matrix, supply, demand):
                st.success("✅ Model built successfully! Go to VAM Solution tab.")
                st.balloons()

with tab2:
    st.markdown('<p class="sub-header">Vogel\'s Approximation Method</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.info("👈 Please build the model first in the Input Data tab.")
    else:
        model = st.session_state.model
        
        # Display current state
        display_cost_matrix(model)
        st.divider()
        display_allocation_matrix(model)
        
        st.divider()
        
        # VAM controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("➡️ Step VAM", use_container_width=True):
                step_data = model.vam_step_forward()
                if step_data:
                    st.session_state.vam_step += 1
                    st.rerun()
                else:
                    st.info("VAM complete!")
        
        with col2:
            if st.button("⏩ Finish VAM", use_container_width=True):
                model.vam_apply_all()
                st.session_state.vam_step = len(model.vam_steps)
                st.success("✅ VAM completed!")
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                model.reset()
                st.session_state.vam_step = 0
                st.session_state.modi_started = False
                st.rerun()
        
        with col4:
            if st.button("⚡ Auto VAM", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_steps = 0
                while not model.vam_is_done():
                    step_data = model.vam_step_forward()
                    if step_data:
                        st.session_state.vam_step += 1
                        total_steps += 1
                        progress_bar.progress(min(total_steps / 20, 1.0))
                        status_text.text(f"Step {st.session_state.vam_step}")
                        time.sleep(0.3)
                    else:
                        break
                
                progress_bar.progress(1.0)
                status_text.text("✅ VAM Complete!")
                st.rerun()
        
        # Show last step info if available
        if st.session_state.vam_step > 0 and st.session_state.vam_step <= len(model.vam_steps):
            st.divider()
            display_vam_step(model.vam_steps[st.session_state.vam_step - 1])

with tab3:
    st.markdown('<p class="sub-header">Modified Distribution (MODI) Method</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.info("👈 Please build the model first in the Input Data tab.")
    elif not st.session_state.model.vam_is_done():
        st.warning("⚠️ Please complete VAM first to get an initial feasible solution.")
    else:
        model = st.session_state.model
        
        # Display allocation
        display_allocation_matrix(model)
        st.divider()
        
        # MODI controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Start MODI", use_container_width=True):
                st.session_state.modi_started = True
                st.rerun()
        
        with col2:
            if st.button("➡️ Step MODI", use_container_width=True) and st.session_state.modi_started:
                pos, val, red = model.find_entering()
                if pos:
                    cycle = model.find_cycle(pos)
                    if cycle:
                        model.apply_cycle(cycle)
                        st.rerun()
                else:
                    st.success("✅ Optimal solution reached!")
        
        with col3:
            if st.button("⏩ Finish MODI", use_container_width=True) and st.session_state.modi_started:
                iterations = 0
                with st.spinner("Optimizing..."):
                    while True:
                        pos, val, red = model.find_entering()
                        if pos is None:
                            break
                        cycle = model.find_cycle(pos)
                        if cycle:
                            model.apply_cycle(cycle)
                            iterations += 1
                        else:
                            break
                st.success(f"✅ Optimal solution reached in {iterations} iterations!")
                st.rerun()
        
        with col4:
            if st.button("🔄 Reset MODI", use_container_width=True):
                st.session_state.modi_started = False
                st.rerun()
        
        # Display UV method results
        if st.session_state.modi_started:
            st.divider()
            has_entering = display_uv_method(model)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p>🏥 Hospital Medical Supplies Transportation Solver</p>
        <p>Created by <strong>Shahem Riaz</strong> | December 2025</p>
        <p>Powered by Streamlit 🎈</p>
    </div>
""", unsafe_allow_html=True)
