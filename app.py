"""
Hospital Medical Supply Transportation Optimization with Big M Method
Streamlit Web Application
File: app.py

Features:
- Standard Transportation Problem (VAM + MODI)
- Big M Method for prohibited routes and special constraints
- Interactive Web Interface

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from copy import deepcopy

# Page configuration
st.set_page_config(
    page_title="Hospital Transportation Optimizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.02);
        transition: all 0.3s ease;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

EPS = 1e-6
BIG_M = 999999  # Big M value for prohibited routes

class TransportationProblemWithBigM:
    def __init__(self, cost, supply, demand, prohibited_routes=None):
        """
        Initialize transportation problem with Big M support
        
        Args:
            cost: m x n cost matrix
            supply: list of supply at each source
            demand: list of demand at each destination
            prohibited_routes: list of tuples (i, j) that are prohibited
        """
        self.original_cost = [list(row) for row in cost]
        self.original_supply = list(supply)
        self.original_demand = list(demand)
        
        # Apply Big M to prohibited routes
        self.cost = deepcopy(self.original_cost)
        self.prohibited_routes = prohibited_routes or []
        
        for i, j in self.prohibited_routes:
            if i < len(self.cost) and j < len(self.cost[0]):
                self.cost[i][j] = BIG_M
        
        self.supply = list(self.original_supply)
        self.demand = list(self.original_demand)
        
        self.m = len(self.supply)
        self.n = len(self.demand)
        
        self.logs = []
        self.balance_problem()
        
        self.m = len(self.supply)
        self.n = len(self.demand)
        
        self.allocation = [[0.0 for _ in range(self.n)] for _ in range(self.m)]
        self.is_basic = [[False for _ in range(self.n)] for _ in range(self.m)]
    
    def log(self, message):
        self.logs.append(message)
    
    def balance_problem(self):
        total_supply = sum(self.supply)
        total_demand = sum(self.demand)
        
        if abs(total_supply - total_demand) < EPS:
            self.log("✅ Problem is balanced")
            return
        
        if total_supply > total_demand:
            diff = total_supply - total_demand
            for row in self.cost:
                row.append(0)
            self.demand.append(diff)
            self.log(f"⚠️ Added dummy hospital with demand {diff:.0f}")
        else:
            diff = total_demand - total_supply
            self.cost.append([0] * len(self.demand))
            self.supply.append(diff)
            self.log(f"⚠️ Added dummy warehouse with supply {diff:.0f}")
    
    def solve_vam(self):
        """VAM with Big M handling"""
        self.log("="*60)
        self.log("📊 Starting VAM (Vogel's Approximation Method)")
        self.log("="*60)
        
        if self.prohibited_routes:
            self.log(f"🚫 Prohibited Routes: {len(self.prohibited_routes)}")
        
        supply = list(self.supply)
        demand = list(self.demand)
        done_row = [False] * self.m
        done_col = [False] * self.n
        step = 1
        
        while any(not d for d in done_row) and any(not d for d in done_col):
            # Calculate row penalties
            row_penalties = []
            for i in range(self.m):
                if done_row[i]:
                    row_penalties.append(-1)
                else:
                    costs = [self.cost[i][j] for j in range(self.n) 
                            if not done_col[j] and self.cost[i][j] < BIG_M]
                    if len(costs) >= 2:
                        costs.sort()
                        row_penalties.append(costs[1] - costs[0])
                    elif len(costs) == 1:
                        row_penalties.append(costs[0])
                    else:
                        row_penalties.append(-1)
            
            # Calculate column penalties
            col_penalties = []
            for j in range(self.n):
                if done_col[j]:
                    col_penalties.append(-1)
                else:
                    costs = [self.cost[i][j] for i in range(self.m) 
                            if not done_row[i] and self.cost[i][j] < BIG_M]
                    if len(costs) >= 2:
                        costs.sort()
                        col_penalties.append(costs[1] - costs[0])
                    elif len(costs) == 1:
                        col_penalties.append(costs[0])
                    else:
                        col_penalties.append(-1)
            
            max_row_pen = max(row_penalties) if row_penalties else -1
            max_col_pen = max(col_penalties) if col_penalties else -1
            
            if max_row_pen == -1 and max_col_pen == -1:
                break
            
            # Select cell based on maximum penalty (avoiding Big M routes)
            if max_row_pen >= max_col_pen:
                i = row_penalties.index(max_row_pen)
                min_cost = float('inf')
                j = -1
                for jj in range(self.n):
                    if not done_col[jj] and self.cost[i][jj] < min_cost:
                        min_cost = self.cost[i][jj]
                        j = jj
            else:
                j = col_penalties.index(max_col_pen)
                min_cost = float('inf')
                i = -1
                for ii in range(self.m):
                    if not done_row[ii] and self.cost[ii][j] < min_cost:
                        min_cost = self.cost[ii][j]
                        i = ii
            
            # Check if we found a valid cell
            if i == -1 or j == -1 or self.cost[i][j] >= BIG_M:
                self.log("⚠️ No valid allocation found, stopping VAM")
                break
            
            qty = min(supply[i], demand[j])
            self.allocation[i][j] = qty
            self.is_basic[i][j] = True
            
            route_status = "🚫 PROHIBITED" if self.cost[i][j] >= BIG_M else ""
            self.log(f"Step {step}: W{i+1}→H{j+1} | {qty:.0f} units @ PKR {self.cost[i][j]:.0f}/unit {route_status}")
            
            supply[i] -= qty
            demand[j] -= qty
            
            if supply[i] < EPS:
                done_row[i] = True
            if demand[j] < EPS:
                done_col[j] = True
            
            step += 1
        
        self.fix_degeneracy()
        cost = self.total_cost()
        self.log(f"\n✅ VAM Complete: Initial Cost = PKR {cost:.2f}")
        return cost
    
    def fix_degeneracy(self):
        target = self.m + self.n - 1
        current = sum(1 for i in range(self.m) for j in range(self.n) if self.is_basic[i][j])
        
        if current < target:
            for i in range(self.m):
                for j in range(self.n):
                    if not self.is_basic[i][j] and current < target and self.cost[i][j] < BIG_M:
                        self.is_basic[i][j] = True
                        self.allocation[i][j] = EPS
                        current += 1
                        if current >= target:
                            break
                if current >= target:
                    break
    
    def compute_uv(self):
        u = [None] * self.m
        v = [None] * self.n
        u[0] = 0
        
        for _ in range(self.m + self.n):
            for i in range(self.m):
                for j in range(self.n):
                    if self.is_basic[i][j]:
                        if u[i] is not None and v[j] is None:
                            v[j] = self.cost[i][j] - u[i]
                        elif v[j] is not None and u[i] is None:
                            u[i] = self.cost[i][j] - v[j]
        
        return u, v
    
    def find_entering_cell(self):
        u, v = self.compute_uv()
        min_reduced = 0
        entering = None
        
        for i in range(self.m):
            for j in range(self.n):
                # Skip prohibited routes and basic variables
                if not self.is_basic[i][j] and self.cost[i][j] < BIG_M:
                    if u[i] is not None and v[j] is not None:
                        reduced_cost = self.cost[i][j] - (u[i] + v[j])
                        if reduced_cost < min_reduced - EPS:
                            min_reduced = reduced_cost
                            entering = (i, j)
        
        return entering, min_reduced
    
    def find_cycle(self, entering):
        basics = [(i, j) for i in range(self.m) for j in range(self.n)
                  if self.is_basic[i][j] or (i, j) == entering]
        basics_set = set(basics)
        
        def get_neighbors(cell, move_row):
            i, j = cell
            neighbors = []
            if move_row:
                for jj in range(self.n):
                    if jj != j and (i, jj) in basics_set:
                        neighbors.append((i, jj))
            else:
                for ii in range(self.m):
                    if ii != i and (ii, j) in basics_set:
                        neighbors.append((ii, j))
            return neighbors
        
        path = [entering]
        visited = {entering}
        
        def backtrack(current, move_row):
            for neighbor in get_neighbors(current, move_row):
                if neighbor == entering and len(path) >= 4:
                    return True
                if neighbor in visited:
                    continue
                
                visited.add(neighbor)
                path.append(neighbor)
                
                if backtrack(neighbor, not move_row):
                    return True
                
                path.pop()
                visited.remove(neighbor)
            
            return False
        
        if backtrack(entering, True):
            return path
        
        visited = {entering}
        path = [entering]
        
        if backtrack(entering, False):
            return path
        
        return None
    
    def apply_cycle(self, cycle):
        minus_positions = [cycle[i] for i in range(1, len(cycle), 2)]
        theta = min(self.allocation[i][j] for i, j in minus_positions)
        
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:
                self.allocation[i][j] += theta
                self.is_basic[i][j] = True
            else:
                self.allocation[i][j] -= theta
                if self.allocation[i][j] < EPS:
                    self.allocation[i][j] = 0
                    self.is_basic[i][j] = False
        
        self.fix_degeneracy()
        return theta
    
    def solve_modi(self):
        self.log("\n" + "="*60)
        self.log("🎯 Starting MODI Method (UV Method)")
        self.log("="*60)
        
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            entering, reduced_cost = self.find_entering_cell()
            
            if entering is None:
                self.log(f"\n✅ Optimal solution reached in {iteration} iterations!")
                break
            
            iteration += 1
            cycle = self.find_cycle(entering)
            
            if cycle is None:
                self.log(f"⚠️ No cycle found at iteration {iteration}")
                self.fix_degeneracy()
                continue
            
            theta = self.apply_cycle(cycle)
            cost = self.total_cost()
            
            self.log(f"Iteration {iteration}: W{entering[0]+1}→H{entering[1]+1} | "
                    f"Reduced: {reduced_cost:.2f} | New Cost: PKR {cost:.2f}")
        
        return iteration
    
    def total_cost(self):
        """Calculate total cost, treating Big M routes as infeasible"""
        total = 0
        has_big_m = False
        
        for i in range(self.m):
            for j in range(self.n):
                if self.allocation[i][j] > EPS:
                    if self.cost[i][j] >= BIG_M:
                        has_big_m = True
                    total += self.allocation[i][j] * self.cost[i][j]
        
        if has_big_m:
            return float('inf')  # Infeasible solution
        
        return total
    
    def get_allocation_df(self):
        data = []
        for i in range(min(self.m, len(self.original_supply))):
            row = [f"W{i+1}"]
            for j in range(min(self.n, len(self.original_demand))):
                val = self.allocation[i][j]
                if self.cost[i][j] >= BIG_M:
                    row.append("🚫")
                else:
                    row.append(int(val) if val > EPS else 0)
            row.append(self.original_supply[i])
            data.append(row)
        
        demand_row = ["Demand"] + list(self.original_demand) + [""]
        data.append(demand_row)
        
        cols = [""] + [f"H{j+1}" for j in range(len(self.original_demand))] + ["Supply"]
        return pd.DataFrame(data, columns=cols)


def generate_random_data(n_warehouses, n_hospitals, seed=42):
    np.random.seed(seed)
    
    cost = np.random.randint(10, 35, size=(n_warehouses, n_hospitals)).tolist()
    
    total = np.random.randint(700, 1000)
    supply = np.random.randint(50, 120, size=n_warehouses)
    supply = (supply / supply.sum() * total).astype(int).tolist()
    
    demand = np.random.randint(50, 120, size=n_hospitals)
    demand = (demand / demand.sum() * total).astype(int).tolist()
    
    diff = sum(supply) - sum(demand)
    if diff > 0:
        demand[0] += diff
    else:
        supply[0] += abs(diff)
    
    return cost, supply, demand


def main():
    # Header
    st.markdown('<div class="main-header">🏥 Hospital Transportation Optimizer with Big M Method</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='font-size: 4rem; margin: 0;'>🏥</h1>
            <h2 style='color: #667eea; margin-top: 0.5rem;'>Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        mode = st.radio("📊 Input Mode", 
                       ["Sample Data", "Custom Problem"],
                       help="Choose between pre-generated sample data or create your own problem")
        
        if mode == "Sample Data":
            n_warehouses = 10
            n_hospitals = 10
            seed = st.number_input("🎲 Random Seed", 1, 1000, 42, help="Change this to generate different random data")
        else:
            n_warehouses = st.number_input("📦 Number of Warehouses", 2, 15, 5)
            n_hospitals = st.number_input("🏥 Number of Hospitals", 2, 15, 5)
            seed = 42
        
        st.markdown("---")
        
        # Big M Configuration
        st.markdown("### 🚫 Big M Method")
        use_big_m = st.checkbox("Enable Prohibited Routes", value=False, 
                               help="Enable this to specify routes that cannot be used")
        
        prohibited_routes = []
        if use_big_m:
            st.info("💡 Select routes that are NOT allowed (e.g., W1→H2 means route from Warehouse 1 to Hospital 2 is prohibited)")
            
            num_prohibited = st.number_input("Number of prohibited routes", 0, min(n_warehouses * n_hospitals, 20), 2)
            
            for idx in range(num_prohibited):
                col1, col2 = st.columns(2)
                with col1:
                    w = st.number_input(f"Route {idx+1}: From W", 1, n_warehouses, min(idx+1, n_warehouses), key=f"w{idx}")
                with col2:
                    h = st.number_input(f"To H", 1, n_hospitals, min(idx+1, n_hospitals), key=f"h{idx}")
                prohibited_routes.append((w-1, h-1))
        
        st.markdown("---")
        st.markdown("### 👥 Project Team")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 8px; color: white;'>
        <b>Section: BSCS-7B</b><br>
        • Yashal Rafique<br>
        • Sahla Farooq<br>
        • Barira Aurangzaib<br>
        • Hafsa Shad<br>
        • Kashifa Kanwal
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    cost, supply, demand = generate_random_data(n_warehouses, n_hospitals, seed)
    
    # Display Input Data
    st.markdown("## 📋 Problem Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📦 Total Supply", f"{sum(supply):,} units", help="Total units available from all warehouses")
    with col2:
        st.metric("🏥 Total Demand", f"{sum(demand):,} units", help="Total units needed by all hospitals")
    with col3:
        st.metric("📐 Problem Size", f"{n_warehouses}×{n_hospitals}", help="Number of warehouses × hospitals")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Warehouse Supply")
        supply_df = pd.DataFrame({
            'Warehouse': [f'W{i+1}' for i in range(len(supply))],
            'Supply (units)': supply
        })
        st.dataframe(supply_df, use_container_width=True, hide_index=True, height=300)
    
    with col2:
        st.markdown("### 🏥 Hospital Demand")
        demand_df = pd.DataFrame({
            'Hospital': [f'H{i+1}' for i in range(len(demand))],
            'Demand (units)': demand
        })
        st.dataframe(demand_df, use_container_width=True, hide_index=True, height=300)
    
    st.markdown("### 💰 Transportation Cost Matrix (PKR per unit)")
    cost_df = pd.DataFrame(cost, 
                          columns=[f'H{i+1}' for i in range(n_hospitals)],
                          index=[f'W{i+1}' for i in range(n_warehouses)])
    
    # Highlight prohibited routes
    if use_big_m and prohibited_routes:
        def highlight_prohibited(row):
            return ['background-color: #ff4444; color: white; font-weight: bold' 
                    if (int(row.name[1:])-1, int(col[1:])-1) in prohibited_routes 
                    else '' 
                    for col in cost_df.columns]
        
        styled_df = cost_df.style.apply(highlight_prohibited, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.warning(f"🚫 {len(prohibited_routes)} prohibited routes marked in red. These routes will not be used in the solution.")
    else:
        st.dataframe(cost_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Solve Button
    if st.button("🚀 Solve Optimization Problem", type="primary"):
        with st.spinner("🔄 Solving transportation problem... Please wait"):
            try:
                problem = TransportationProblemWithBigM(cost, supply, demand, prohibited_routes if use_big_m else None)
                
                initial_cost = problem.solve_vam()
                iterations = problem.solve_modi()
                final_cost = problem.total_cost()
                
                if final_cost == float('inf'):
                    st.error("❌ No feasible solution exists! The problem is infeasible with the current constraints. Try removing some prohibited routes.")
                    st.stop()
                
                st.success("✅ Optimization Complete!")
                
                # Metrics
                st.markdown("## 📊 Solution Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Initial Cost (VAM)", f"PKR {initial_cost:,.2f}", 
                             help="Initial solution cost using Vogel's Approximation Method")
                
                with col2:
                    st.metric("Final Cost (MODI)", f"PKR {final_cost:,.2f}", 
                             help="Optimized cost using Modified Distribution Method")
                
                with col3:
                    improvement = initial_cost - final_cost
                    pct = (improvement/initial_cost*100) if initial_cost > 0 else 0
                    st.metric("Cost Saved", f"PKR {improvement:,.2f}", 
                             delta=f"-{pct:.1f}%", delta_color="inverse",
                             help="Total cost reduction achieved through optimization")
                
                with col4:
                    st.metric("MODI Iterations", iterations,
                             help="Number of iterations taken to reach optimal solution")
                
                st.markdown("---")
                
                # Allocation Table
                st.markdown("### 📋 Optimal Allocation Matrix")
                st.markdown("*Shows the quantity of supplies to transport from each warehouse to each hospital*")
                allocation_df = problem.get_allocation_df()
                st.dataframe(allocation_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Route Details
                st.markdown("### 🚚 Active Transportation Routes")
                routes = []
                for i in range(min(problem.m, len(supply))):
                    for j in range(min(problem.n, len(demand))):
                        if problem.allocation[i][j] > EPS and problem.cost[i][j] < BIG_M:
                            routes.append({
                                'Route': f'W{i+1} → H{j+1}',
                                'Units': int(problem.allocation[i][j]),
                                'Cost/Unit (PKR)': problem.cost[i][j],
                                'Total Cost (PKR)': int(problem.allocation[i][j] * problem.cost[i][j])
                            })
                
                if routes:
                    routes_df = pd.DataFrame(routes)
                    routes_df = routes_df.sort_values('Total Cost (PKR)', ascending=False)
                    st.dataframe(routes_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    st.markdown("### 📊 Cost Breakdown Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(routes_df, x='Route', y='Total Cost (PKR)', 
                                    title='Transportation Cost by Route',
                                    color='Total Cost (PKR)',
                                    color_continuous_scale='Viridis',
                                    labels={'Total Cost (PKR)': 'Cost (PKR)'})
                        fig.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(routes_df, values='Units', names='Route',
                                    title='Distribution of Units by Route',
                                    color_discrete_sequence=px.colors.qualitative.Set3)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No active routes found in the solution.")
                
                # Solution Log
                st.markdown("---")
                with st.expander("📝 View Detailed Solution Log"):
                    st.code("\n".join(problem.logs), language="text")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
