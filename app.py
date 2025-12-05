"""
Hospital Medical Supply Transportation Optimization with Big M Method
Google Colab Version
Simple Console-Based Implementation

Features:
- Standard Transportation Problem (VAM + MODI)
- Big M Method for prohibited routes
- Console-based interface with formatted output
"""

import numpy as np
import pandas as pd
from copy import deepcopy

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
        print(message)
    
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
        self.log("\n" + "="*70)
        self.log("📊 STARTING VAM (VOGEL'S APPROXIMATION METHOD)")
        self.log("="*70)
        
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
        self.log(f"\n✅ VAM Complete: Initial Cost = PKR {cost:,.2f}\n")
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
        self.log("="*70)
        self.log("🎯 STARTING MODI METHOD (UV METHOD)")
        self.log("="*70)
        
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
    
    def print_solution(self):
        """Print the solution in a formatted way"""
        print("\n" + "="*70)
        print("📋 OPTIMAL ALLOCATION MATRIX")
        print("="*70)
        
        # Create allocation table
        headers = [""] + [f"H{j+1}" for j in range(len(self.original_demand))] + ["Supply"]
        
        print(f"{'':>8}", end="")
        for header in headers[1:-1]:
            print(f"{header:>8}", end="")
        print(f"{headers[-1]:>10}")
        
        print("-" * 70)
        
        for i in range(min(self.m, len(self.original_supply))):
            print(f"W{i+1:>6} |", end="")
            for j in range(min(self.n, len(self.original_demand))):
                val = self.allocation[i][j]
                if self.cost[i][j] >= BIG_M:
                    print(f"{'🚫':>8}", end="")
                else:
                    print(f"{int(val):>8}", end="") if val > EPS else print(f"{0:>8}", end="")
            print(f"{self.original_supply[i]:>10}")
        
        print("-" * 70)
        print(f"{'Demand':>8}", end="")
        for d in self.original_demand:
            print(f"{d:>8}", end="")
        print()
        
        # Print active routes
        print("\n" + "="*70)
        print("🚚 ACTIVE TRANSPORTATION ROUTES")
        print("="*70)
        
        routes = []
        for i in range(min(self.m, len(self.original_supply))):
            for j in range(min(self.n, len(self.original_demand))):
                if self.allocation[i][j] > EPS and self.cost[i][j] < BIG_M:
                    routes.append({
                        'from': f'W{i+1}',
                        'to': f'H{j+1}',
                        'units': int(self.allocation[i][j]),
                        'cost_per_unit': self.cost[i][j],
                        'total_cost': int(self.allocation[i][j] * self.cost[i][j])
                    })
        
        routes.sort(key=lambda x: x['total_cost'], reverse=True)
        
        print(f"{'Route':>15} | {'Units':>10} | {'Cost/Unit':>12} | {'Total Cost':>15}")
        print("-" * 70)
        
        for route in routes:
            print(f"{route['from']} → {route['to']:>6} | {route['units']:>10} | "
                  f"PKR {route['cost_per_unit']:>7.0f} | PKR {route['total_cost']:>11,}")
        
        print("-" * 70)
        print(f"{'TOTAL':>31} | {'':>12} | PKR {sum(r['total_cost'] for r in routes):>11,}")


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


def main():
    """Main function to run the optimization"""
    print("\n" + "="*70)
    print("🏥 HOSPITAL MEDICAL SUPPLY TRANSPORTATION OPTIMIZER")
    print("="*70)
    print("\nProject Team - Section BSCS-7B:")
    print("  • Yashal Rafique")
    print("  • Sahla Farooq")
    print("  • Barira Aurangzaib")
    print("  • Hafsa Shad")
    print("  • Kashifa Kanwal")
    print("="*70)
    
    # Generate sample data
    print("\n📊 Generating sample problem data...")
    n_warehouses = 5
    n_hospitals = 5
    cost, supply, demand = generate_sample_data(n_warehouses, n_hospitals, seed=42)
    
    # Define prohibited routes (optional)
    prohibited_routes = [
        (0, 1),  # W1 → H2 is prohibited
        (2, 3),  # W3 → H4 is prohibited
    ]
    
    print(f"\n✓ Problem Size: {n_warehouses} Warehouses × {n_hospitals} Hospitals")
    print(f"✓ Total Supply: {sum(supply):,} units")
    print(f"✓ Total Demand: {sum(demand):,} units")
    print(f"✓ Prohibited Routes: {len(prohibited_routes)}")
    
    # Create and solve problem
    print("\n" + "="*70)
    problem = TransportationProblemWithBigM(cost, supply, demand, prohibited_routes)
    
    # Solve using VAM
    initial_cost = problem.solve_vam()
    
    # Optimize using MODI
    iterations = problem.solve_modi()
    
    # Get final cost
    final_cost = problem.total_cost()
    
    # Check feasibility
    if final_cost == float('inf'):
        print("\n❌ ERROR: No feasible solution exists!")
        print("The problem is infeasible with current constraints.")
        return
    
    # Print results
    print("\n" + "="*70)
    print("📊 OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Initial Cost (VAM)    : PKR {initial_cost:>15,.2f}")
    print(f"Final Cost (MODI)     : PKR {final_cost:>15,.2f}")
    print(f"Cost Saved            : PKR {initial_cost - final_cost:>15,.2f}")
    print(f"Improvement           : {((initial_cost - final_cost) / initial_cost * 100):>14.2f}%")
    print(f"MODI Iterations       : {iterations:>19}")
    
    # Print solution
    problem.print_solution()
    
    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
