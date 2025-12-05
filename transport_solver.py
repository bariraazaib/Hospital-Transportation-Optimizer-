"""
Transportation Problem Solver - Core Logic
VAM + MODI/UV Method Implementation

Author: Barira zaib
Date: 2025-12-05
"""

from copy import deepcopy

EPS = 1e-9

class TransportModel:
    """
    Transportation problem solver using VAM for initial solution
    and MODI/UV method for optimization
    """
    
    def __init__(self, cost, supply, demand, verbose=False):
        """
        Initialize transportation model
        
        Args:
            cost: 2D list of costs (m x n)
            supply: list of supply values (length m)
            demand: list of demand values (length n)
            verbose: print debug information
        """
        self.user_cost = [list(map(float, row)) for row in cost]
        self.user_supply = list(map(float, supply))
        self.user_demand = list(map(float, demand))
        self.verbose = verbose
        self.reset()

    def reset(self):
        """Initialize internal balanced problem and allocations"""
        self.cost = deepcopy(self.user_cost)
        self.supply = list(self.user_supply)
        self.demand = list(self.user_demand)
        self.m = len(self.supply)
        self.n = len(self.demand)
        self.dummy_added = None  # 'row' or 'col' or None
        self._balance()
        self.m = len(self.supply)
        self.n = len(self.demand)
        self.alloc = [[0.0 for _ in range(self.n)] for _ in range(self.m)]
        self.is_basic = [[False for _ in range(self.n)] for _ in range(self.m)]
        
        # VAM state
        self.vam_steps = []
        self.vam_state_index = -1
        self._prepare_vam()
        
        # MODI/UV state
        self.u = [None] * self.m
        self.v = [None] * self.n
        self.modi_steps = []
        self.modi_index = -1

    def _balance(self):
        """Balance supply and demand by adding dummy row/column"""
        total_supply = sum(self.supply)
        total_demand = sum(self.demand)
        
        if abs(total_supply - total_demand) < EPS:
            self.dummy_added = None
            return
        
        if total_supply > total_demand:
            # Add dummy demand column
            diff = total_supply - total_demand
            for row in self.cost:
                row.append(0.0)
            self.demand.append(diff)
            self.dummy_added = 'col'
        else:
            # Add dummy supply row
            diff = total_demand - total_supply
            self.cost.append([0.0] * len(self.demand))
            self.supply.append(diff)
            self.dummy_added = 'row'

    def _prepare_vam(self):
        """
        Prepare VAM steps for step-by-step execution
        Each step allocates to one cell
        """
        supply = list(self.supply)
        demand = list(self.demand)
        m, n = self.m, self.n
        done_row = [False] * m
        done_col = [False] * n
        remaining_rows = m
        remaining_cols = n
        cost = deepcopy(self.cost)

        def row_penalty(i):
            """Calculate penalty for row i"""
            vals = [cost[i][j] for j in range(n) if not done_col[j]]
            if not vals:
                return -1
            if len(vals) == 1:
                return vals[0]
            s = sorted(vals)
            return s[1] - s[0]

        def col_penalty(j):
            """Calculate penalty for column j"""
            vals = [cost[i][j] for i in range(m) if not done_row[i]]
            if not vals:
                return -1
            if len(vals) == 1:
                return vals[0]
            s = sorted(vals)
            return s[1] - s[0]

        steps = []
        
        while remaining_rows > 0 and remaining_cols > 0:
            # Calculate penalties
            row_pen = [row_penalty(i) if not done_row[i] else -1 for i in range(m)]
            col_pen = [col_penalty(j) if not done_col[j] else -1 for j in range(n)]
            
            # Choose maximum penalty
            max_row_pen = max(row_pen)
            max_col_pen = max(col_pen)
            
            if max_row_pen == -1 and max_col_pen == -1:
                break
            
            if max_row_pen >= max_col_pen:
                i = row_pen.index(max_row_pen)
                # Select cheapest available column in row i
                candidates = [(cost[i][j], j) for j in range(n) if not done_col[j]]
                _, j = min(candidates)
                chosen_type = 'row'
            else:
                j = col_pen.index(max_col_pen)
                # Select cheapest available row in column j
                candidates = [(cost[i][j], i) for i in range(m) if not done_row[i]]
                _, i = min(candidates)
                chosen_type = 'col'
            
            # Allocate
            q = min(supply[i], demand[j])
            
            steps.append({
                'type': chosen_type,
                'penalty_row': row_pen,
                'penalty_col': col_pen,
                'chosen_cell': (i, j),
                'alloc_qty': q,
                'supply_before': supply[:],
                'demand_before': demand[:],
                'cost_cell': cost[i][j]
            })
            
            # Update
            supply[i] -= q
            demand[j] -= q
            
            if abs(supply[i]) < EPS:
                done_row[i] = True
                remaining_rows -= 1
            if abs(demand[j]) < EPS:
                done_col[j] = True
                remaining_cols -= 1
        
        self.vam_steps = steps
        self.vam_state_index = -1

    def vam_step_forward(self):
        """
        Apply next VAM step (one allocation)
        
        Returns:
            dict: Step information or None if finished
        """
        if self.vam_state_index + 1 >= len(self.vam_steps):
            return None
        
        self.vam_state_index += 1
        st = self.vam_steps[self.vam_state_index]
        i, j = st['chosen_cell']
        q = st['alloc_qty']
        
        # Allocate
        self.alloc[i][j] += q
        self.is_basic[i][j] = True
        
        # Update internal supply/demand
        self.supply = [stv for stv in st['supply_before']]
        self.demand = [dtv for dtv in st['demand_before']]
        self.supply[i] -= q
        self.demand[j] -= q
        
        return st

    def vam_is_done(self):
        """Check if VAM is complete"""
        return self.vam_state_index + 1 >= len(self.vam_steps)

    def vam_apply_all(self):
        """Apply all remaining VAM steps"""
        while not self.vam_is_done():
            self.vam_step_forward()
        self._fix_degeneracy()

    def _fix_degeneracy(self):
        """Ensure m+n-1 basic variables"""
        target = self.m + self.n - 1
        current = sum(1 for i in range(self.m) for j in range(self.n) 
                     if self.is_basic[i][j])
        
        if current >= target:
            return
        
        # Add tiny allocations to non-basic cells
        for i in range(self.m):
            for j in range(self.n):
                if not self.is_basic[i][j]:
                    self.is_basic[i][j] = True
                    self.alloc[i][j] = EPS
                    current += 1
                    if current >= target:
                        return

    def compute_uv(self):
        """
        Compute u and v potentials for current basic cells
        
        Returns:
            tuple: (u, v) lists of potentials
        """
        m, n = self.m, self.n
        u = [None] * m
        v = [None] * n
        
        # Set u[0] = 0
        u[0] = 0.0
        
        # Iteratively compute u and v
        for _ in range(m + n):
            updated = False
            for i in range(m):
                for j in range(n):
                    if self.is_basic[i][j]:
                        if u[i] is not None and v[j] is None:
                            v[j] = self.cost[i][j] - u[i]
                            updated = True
                        elif v[j] is not None and u[i] is None:
                            u[i] = self.cost[i][j] - v[j]
                            updated = True
        
        self.u = u
        self.v = v
        return u, v

    def reduced_costs(self):
        """
        Calculate reduced costs for all cells
        
        Returns:
            list: 2D list of reduced costs (c_ij - u_i - v_j)
        """
        u, v = self.compute_uv()
        red = [[None] * self.n for _ in range(self.m)]
        
        for i in range(self.m):
            for j in range(self.n):
                if u[i] is None or v[j] is None:
                    red[i][j] = None
                else:
                    red[i][j] = self.cost[i][j] - (u[i] + v[j])
        
        return red

    def find_entering(self):
        """
        Find the most negative reduced cost cell (entering variable)
        
        Returns:
            tuple: (position, value, reduced_costs_matrix)
        """
        red = self.reduced_costs()
        min_val = 0
        pos = None
        
        for i in range(self.m):
            for j in range(self.n):
                if not self.is_basic[i][j] and red[i][j] is not None:
                    if red[i][j] < min_val - EPS:
                        min_val = red[i][j]
                        pos = (i, j)
        
        return pos, min_val, red

    def find_cycle(self, start):
        """
        Find alternating cycle including start (entering cell)
        
        Args:
            start: tuple (i, j) of entering cell
            
        Returns:
            list: List of positions forming the cycle or None
        """
        m, n = self.m, self.n
        basics = [(i, j) for i in range(m) for j in range(n) 
                 if self.is_basic[i][j] or (i, j) == start]
        basics_set = set(basics)

        def neighbors(cell, move_row):
            """Get neighbors alternating between row and column moves"""
            i, j = cell
            res = []
            if move_row:
                for jj in range(n):
                    if jj != j and (i, jj) in basics_set:
                        res.append((i, jj))
            else:
                for ii in range(m):
                    if ii != i and (ii, j) in basics_set:
                        res.append((ii, j))
            return res

        path = [start]
        visited = set([start])

        def backtrack(curr, must_move_row):
            """Backtracking to find cycle"""
            for nb in neighbors(curr, must_move_row):
                if nb == start and len(path) >= 4:
                    return True
                if nb in visited:
                    continue
                visited.add(nb)
                path.append(nb)
                if backtrack(nb, not must_move_row):
                    return True
                path.pop()
                visited.remove(nb)
            return False

        # Try both parity starts
        if backtrack(start, True):
            return path[:]
        
        visited = set([start])
        path = [start]
        
        if backtrack(start, False):
            return path[:]
        
        return None

    def apply_cycle(self, cycle):
        """
        Apply theta adjustment along cycle
        
        Args:
            cycle: list of positions forming the cycle
            
        Returns:
            tuple: (minus_positions, theta)
        """
        minus_positions = [cycle[i] for i in range(1, len(cycle), 2)]
        theta = min(self.alloc[i][j] for (i, j) in minus_positions)
        
        # Adjust allocations
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:  # Plus
                self.alloc[i][j] += theta
                self.is_basic[i][j] = True
            else:  # Minus
                self.alloc[i][j] -= theta
                if self.alloc[i][j] <= EPS:
                    self.alloc[i][j] = 0.0
                    self.is_basic[i][j] = False
        
        self._fix_degeneracy()
        return minus_positions, theta

    def total_cost(self):
        """
        Calculate total transportation cost
        
        Returns:
            float: Total cost
        """
        return sum(self.alloc[i][j] * self.cost[i][j] 
                  for i in range(self.m) for j in range(self.n))
