"""
Hospital Medical Supply Transportation Optimizer - Streamlit Web App
Developed by: BSCS-7B Team
"""

import streamlit as st
import numpy as np
import pandas as pd
from copy import deepcopy
import io

# Constants
EPS = 1e-6
BIG_M = 999999

class TransportationProblemWithBigM:
    def __init__(self, cost, supply, demand, prohibited_routes=None):
        """Initialize transportation problem with Big M support"""
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
        """Balance supply and demand"""
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
            self.log(f"⚠️  Added dummy hospital with demand {diff:.0f}")
        else:
            diff = total_demand - total_supply
            self.cost.append([0] * len(self.demand))
            self.supply.append(diff)
            self.log(f"⚠️  Added dummy warehouse with supply {diff:.0f}")
    
    def solve_vam(self):
        """Vogel's Approximation Method with Big M handling"""
        self.log("\n📊 STARTING VAM (VOGEL'S APPROXIMATION METHOD)")
        
        if self.prohibited_routes:
            self.log(f"🚫 Number of Prohibited Routes: {len(self.prohibited_routes)}")
            for i, j in self.prohibited_routes:
                self.log(f"   - Route W{i+1} → H{j+1} is PROHIBITED")
        
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
            
            # Select cell based on maximum penalty
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
            
            if i == -1 or j == -1 or self.cost[i][j] >= BIG_M:
                self.log("⚠️  No valid allocation found, stopping VAM")
                break
            
            qty = min(supply[i], demand[j])
            self.allocation[i][j] = qty
            self.is_basic[i][j] = True
            
            self.log(f"Step {step:2d}: W{i+1} → H{j+1} | Allocate {qty:5.0f} units @ PKR {self.cost[i][j]:5.0f}/unit")
            
            supply[i] -= qty
            demand[j] -= qty
            
            if supply[i] < EPS:
                done_row[i] = True
            if demand[j] < EPS:
                done_col[j] = True
            
            step += 1
        
        self.fix_degeneracy()
        cost = self.total_cost()
        self.log(f"\n✅ VAM Complete: Initial Cost = PKR {cost:,.2f}")
        return cost
    
    def fix_degeneracy(self):
        """Fix degeneracy by adding epsilon allocations"""
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
        """Compute u and v values for MODI method"""
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
        """Find the entering cell with most negative reduced cost"""
        u, v = self.compute_uv()
        min_reduced = 0
        entering = None
        
        for i in range(self.m):
            for j in range(self.n):
                if not self.is_basic[i][j] and self.cost[i][j] < BIG_M:
                    if u[i] is not None and v[j] is not None:
                        reduced_cost = self.cost[i][j] - (u[i] + v[j])
                        if reduced_cost < min_reduced - EPS:
                            min_reduced = reduced_cost
                            entering = (i, j)
        
        return entering, min_reduced
    
    def find_cycle(self, entering):
        """Find a cycle starting from the entering cell"""
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
        """Apply the cycle to improve the solution"""
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
        """MODI (Modified Distribution) Method"""
        self.log("\n🎯 STARTING MODI METHOD (UV METHOD)")
        
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
                self.log(f"⚠️  No cycle found at iteration {iteration}")
                self.fix_degeneracy()
                continue
            
            theta = self.apply_cycle(cycle)
            cost = self.total_cost()
            
            self.log(f"Iteration {iteration:2d}: Entering Cell W{entering[0]+1}→H{entering[1]+1} | "
                    f"Reduced Cost: {reduced_cost:7.2f} | New Total Cost: PKR {cost:,.2f}")
        
        return iteration
    
    def total_cost(self):
        """Calculate total transportation cost"""
        total = 0
        has_big_m = False
        
        for i in range(self.m):
            for j in range(self.n):
                if self.allocation[i][j] > EPS:
                    if self.cost[i][j] >= BIG_M:
                        has_big_m = True
                    total += self.allocation[i][j] * self.cost[i][j]
        
        if has_big_m:
            return float('inf')
        
        return total
    
    def get_allocation_dataframe(self):
        """Get allocation as pandas DataFrame"""
        headers = [f"H{j+1}" for j in range(len(self.original_demand))] + ["Supply"]
        rows = []
        
        for i in range(min(self.m, len(self.original_supply))):
            row = [f"W{i+1}"]
            for j in range(min(self.n, len(self.original_demand))):
                val = self.allocation[i][j]
                if self.cost[i][j] >= BIG_M:
                    row.append("🚫")
                else:
                    row.append(int(val) if val > EPS else 0)
            row.append(self.original_supply[i])
            rows.append(row)
        
        demand_row = ["Demand"] + self.original_demand + [""]
        rows.append(demand_row)
        
        df = pd.DataFrame(rows, columns=[""] + headers)
        return df
    
    def get_routes_dataframe(self):
        """Get active routes as pandas DataFrame"""
        routes = []
        for i in range(min(self.m, len(self.original_supply))):
            for j in range(min(self.n, len(self.original_demand))):
                if self.allocation[i][j] > EPS and self.cost[i][j] < BIG_M:
                    routes.append({
                        'Route': f'W{i+1} → H{j+1}',
                        'Units': int(self.allocation[i][j]),
                        'Cost/Unit (PKR)': int(self.cost[i][j]),
                        'Total Cost (PKR)': int(self.allocation[i][j] * self.cost[i][j])
                    })
        
        if routes:
            df = pd.DataFrame(routes)
            df = df.sort_values('Total Cost (PKR)', ascending=False)
            return df
        return pd.DataFrame()

def generate_sample_data(n_warehouses=5, n_hospitals=5, seed=42):
    """Generate sample data for testing"""
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

# Streamlit App
def main():
    st.set_page_config(
        page_title="Hospital Supply Optimizer",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Hospital Medical Supply Transportation Optimizer")
    st.markdown("### Using VAM + MODI Method with Big M")
    
    st.markdown("""
    **Developed by BSCS-7B Team:**
    - Yashal Rafique
    - Sahla Farooq
    - Barira Aurangzaib
    - Hafsa Shad
    - Kashifa Kanwal
    """)
    
    st.divider()
    
    # Sidebar for input
    with st.sidebar:
        st.header("⚙️ Problem Configuration")
        
        input_method = st.radio(
            "Input Method:",
            ["Use Sample Data", "Manual Input"]
        )
        
        if input_method == "Use Sample Data":
            n_warehouses = st.slider("Number of Warehouses", 2, 8, 5)
            n_hospitals = st.slider("Number of Hospitals", 2, 8, 5)
            seed = st.number_input("Random Seed", 1, 100, 42)
            
            cost, supply, demand = generate_sample_data(n_warehouses, n_hospitals, seed)
            
            st.success(f"✓ Generated {n_warehouses}×{n_hospitals} problem")
            
        else:
            n_warehouses = st.number_input("Number of Warehouses", 2, 10, 3)
            n_hospitals = st.number_input("Number of Hospitals", 2, 10, 3)
            
            st.subheader("Supply")
            supply = []
            for i in range(n_warehouses):
                val = st.number_input(f"W{i+1} Supply", 0, 1000, 100, key=f"supply_{i}")
                supply.append(val)
            
            st.subheader("Demand")
            demand = []
            for j in range(n_hospitals):
                val = st.number_input(f"H{j+1} Demand", 0, 1000, 100, key=f"demand_{j}")
                demand.append(val)
            
            st.subheader("Cost Matrix (PKR)")
            cost = []
            for i in range(n_warehouses):
                row = []
                cols = st.columns(n_hospitals)
                for j, col in enumerate(cols):
                    val = col.number_input(f"W{i+1}→H{j+1}", 0, 100, 20, key=f"cost_{i}_{j}", label_visibility="collapsed")
                    row.append(val)
                cost.append(row)
        
        st.divider()
        
        st.subheader("🚫 Prohibited Routes")
        n_prohibited = st.number_input("Number of Prohibited Routes", 0, 10, 0)
        
        prohibited_routes = []
        for k in range(n_prohibited):
            col1, col2 = st.columns(2)
            with col1:
                i = st.selectbox(f"From W", range(1, n_warehouses+1), key=f"from_{k}") - 1
            with col2:
                j = st.selectbox(f"To H", range(1, n_hospitals+1), key=f"to_{k}") - 1
            prohibited_routes.append((i, j))
        
        solve_button = st.button("🚀 Solve Problem", type="primary", use_container_width=True)
    
    # Main content
    if solve_button:
        with st.spinner("Solving transportation problem..."):
            try:
                # Create problem
                problem = TransportationProblemWithBigM(cost, supply, demand, prohibited_routes)
                
                # Solve
                initial_cost = problem.solve_vam()
                iterations = problem.solve_modi()
                final_cost = problem.total_cost()
                
                # Check feasibility
                if final_cost == float('inf'):
                    st.error("❌ No feasible solution exists with current constraints!")
                    st.stop()
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Initial Cost (VAM)", f"PKR {initial_cost:,.0f}")
                
                with col2:
                    st.metric("Final Cost (MODI)", f"PKR {final_cost:,.0f}")
                
                with col3:
                    savings = initial_cost - final_cost
                    st.metric("Cost Saved", f"PKR {savings:,.0f}")
                
                with col4:
                    improvement = (savings / initial_cost * 100) if initial_cost > 0 else 0
                    st.metric("Improvement", f"{improvement:.2f}%")
                
                st.divider()
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📋 Allocation Matrix", "🚚 Active Routes", "📝 Optimization Log"])
                
                with tab1:
                    st.subheader("Optimal Allocation Matrix")
                    df_allocation = problem.get_allocation_dataframe()
                    st.dataframe(df_allocation, use_container_width=True, hide_index=True)
                
                with tab2:
                    st.subheader("Active Transportation Routes")
                    df_routes = problem.get_routes_dataframe()
                    if not df_routes.empty:
                        st.dataframe(df_routes, use_container_width=True, hide_index=True)
                        
                        total_cost = df_routes['Total Cost (PKR)'].sum()
                        st.markdown(f"**Total Transportation Cost: PKR {total_cost:,}**")
                        
                        # Chart
                        st.bar_chart(df_routes.set_index('Route')['Total Cost (PKR)'])
                    else:
                        st.info("No active routes found.")
                
                with tab3:
                    st.subheader("Optimization Process Log")
                    log_text = "\n".join(problem.logs)
                    st.text_area("Process Log", log_text, height=400)
                
                st.success("✅ Optimization Complete!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    else:
        st.info("👈 Configure the problem in the sidebar and click 'Solve Problem' to start optimization.")
        
        # Show example
        st.subheader("📊 About This Tool")
        st.markdown("""
        This tool solves the **Transportation Problem** using:
        - **VAM (Vogel's Approximation Method)**: Finds initial feasible solution
        - **MODI Method (UV Method)**: Optimizes the solution to minimum cost
        - **Big M Method**: Handles prohibited routes
        
        **Use Cases:**
        - Medical supply distribution to hospitals
        - Warehouse to retail store logistics
        - Any supply-demand optimization problem
        """)

if __name__ == "__main__":
    main()
