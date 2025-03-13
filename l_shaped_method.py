"""
This module implements the L-shaped method algorithm with detailed output.

When it says "IF VERBOSE" that checks to see whether or not the problem should print... it's declared in main

"""
import pyomo.environ as pyo
from master_problem import solve_master_problem
from subproblems import solve_all_subproblems
from stochastic_problem import get_problem_data, calculate_rhs

def print_master_formulation(c, A, b, x_lb, cuts, iteration):
    """
    Print the formulation of the master problem.
    """
    print(f"\nMaster Problem Formulation (Iteration {iteration}):")
    print("=" * 80)
    
    # Objective function
    obj_terms = [f"{c[i]}*x{i+1}" for i in range(len(c))]
    if cuts:
        obj_str = f"Minimize  {' + '.join(obj_terms)} + θ"
    else:
        obj_str = f"Minimize  {' + '.join(obj_terms)}"
    print(obj_str)
    
    # Constraints
    print("\nSubject to:")
    for i, row in enumerate(A):
        constr_terms = [f"{row[j]}*x{j+1}" for j in range(len(row)) if row[j] != 0]
        constr_str = f"  {' + '.join(constr_terms)} ≤ {b[i]}"
        print(constr_str)
    
    # Variable bounds
    for i, lb in enumerate(x_lb):
        print(f"  x{i+1} ≥ {lb}")
    
    # Optimality cuts
    if cuts:
        print("\nOptimality cuts:")
        for i, (E, e) in enumerate(cuts):
            cut_terms = [f"{E[j]:.4f}*x{j+1}" for j in range(len(E)) if abs(E[j]) > 1e-10]
            if cut_terms:
                cut_str = f"  {' + '.join(cut_terms)} + θ ≥ {e:.4f}"
            else:
                cut_str = f"  θ ≥ {e:.4f}"
            print(cut_str)
    
    print("=" * 80)

def print_subproblem_formulation(x, scenario, result, scenario_index):
    """
    Print the formulation and solution of a subproblem.
    """
    prob = scenario['probability']
    q1 = scenario['q1']
    q2 = scenario['q2']
    d1 = scenario['d1']
    d2 = scenario['d2']
    h = scenario['h']
    T = scenario['T']
    
    # Calculate RHS values
    rhs = calculate_rhs(h, T, x)
    
    print(f"\nSubproblem Formulation (Scenario {scenario_index}, Probability = {prob}):")
    print("=" * 80)
    
    # Print the matrices
    print("h vector:")
    print(f"  h = {h}")
    
    print("\nT matrix:")
    for i, row in enumerate(T):
        print(f"  T[{i}] = {row}")
    
    # Objective
    print(f"\nMinimize  {q1}*y1 + {q2}*y2")
    
    # Constraints
    print("\nSubject to:")
    print(f"  6*y1 + 10*y2 ≤ {rhs[0]:.4f}  (= {h[0]} - {T[0][0]}*{x[0]} - {T[0][1]}*{x[1]})")
    print(f"  8*y1 + 5*y2 ≤ {rhs[1]:.4f}  (= {h[1]} - {T[1][0]}*{x[0]} - {T[1][1]}*{x[1]})")
    print(f"  0 ≤ y1 ≤ {d1}")
    print(f"  0 ≤ y2 ≤ {d2}")
    
    # Solution
    print("\nSolution:")
    print(f"  y1 = {result['y_values'][0]:.4f}")
    print(f"  y2 = {result['y_values'][1]:.4f}")
    print(f"  Objective value = {result['objective_value']:.4f}")
    
    # Dual values (π vector)
    print("\nDual values (π vector):")
    print(f"  π = {result['pi_vector']}")
    
    # Calculate e = π^T * h manually to verify
    manual_e = sum(result['pi_vector'][i] * h[i] for i in range(len(h)))
    manual_e *= prob
    
    # Cut components
    print("\nCut component calculations:")
    print(f"  e = p * π^T * h = {prob} * ({' + '.join([f'({result['pi_vector'][i]}) * ({h[i]})' for i in range(len(h))])}) = {prob} * {manual_e/prob:.4f} = {manual_e:.4f}")
    print(f"  Final e = {result['e']:.4f}")  # Show the actual e used (may include adjustments)
    print(f"  E = p * π^T * T = {prob} * π^T * T = [{', '.join([f'{E:.4f}' for E in result['E']])}]")
    print(f"  w = e - E*x = {result['e']:.4f} - E*{x} = {result['w']:.4f}")
    
    print("=" * 80)

def print_combined_results(results, iteration):
    """
    Print the combined results from all subproblems.
    """
    print(f"\nCombined Results (Iteration {iteration}):")
    print("=" * 80)
    print(f"Total objective value: {results['total_objective']:.4f}")
    print(f"Total e: {results['total_e']:.4f}")
    print(f"Total E: [{', '.join([f'{E:.4f}' for E in results['total_E']])}]")
    print(f"Total w: {results['total_w']:.4f}")
    print("=" * 80)

def run_l_shaped_method(max_iterations=100, tolerance=1e-6, verbose=True):
    """
    Implement the L-shaped method for two-stage stochastic programming with detailed output.
    
    Parameters:
    -----------
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict
        Final solution and performance metrics
    """
    # Get problem data
    data = get_problem_data()
    c = data['c']
    A = data['A']
    b = data['b']
    x_lb = data['x_lb']
    scenarios = data['scenarios']
    
    # Initialize
    cuts = []
    iterations = 0
    converged = False
    history = []
    
    if verbose:
        print("\n" + "="*80)
        print("STARTING L-SHAPED METHOD")
        print("="*80)
    
    while not converged and iterations < max_iterations:
        iterations += 1
        
        # Step 1: Solve master problem
        if verbose:
            print(f"\nITERATION {iterations}")
            print("-" * 80)
            print(f"Step 1: Solving master problem")
            # Print master problem formulation
            print_master_formulation(c, A, b, x_lb, cuts, iterations)
        
        master_result = solve_master_problem(c, A, b, x_lb, cuts)
        x = master_result['x']
        theta = master_result['theta']
        
        if verbose:
            print(f"Master problem solution:")
            print(f"  x = [{', '.join([f'{xi:.6f}' for xi in x])}]")
            print(f"  θ = {theta:.6f}")
            print(f"  Master objective = {master_result['objective']:.6f}")
        
        # Step 3: Solve subproblems
        if verbose:
            print(f"\nStep 3: Solving subproblems")
        
        sub_results = solve_all_subproblems(x, scenarios, iterations)
        w = sub_results['total_w']
        
        # Print detailed subproblem information
        if verbose:
            for i, (scenario, result) in enumerate(zip(scenarios, sub_results['scenario_results'])):
                print_subproblem_formulation(x, scenario, result, i+1)
            
            # Print combined results
            print_combined_results(sub_results, iterations)
        
        # Store iteration history
        history.append({
            'iteration': iterations,
            'x': x,
            'theta': theta,
            'w': w,
            'gap': w - theta,
            'master_objective': master_result['objective'],
            'recourse_objective': sub_results['total_objective']
        })
        
        # Check if this cut would be a duplicate
        is_duplicate = False
        if cuts:
            for E_prev, e_prev in cuts:
                if (abs(sub_results['total_E'][0] - E_prev[0]) < 1e-6 and 
                    abs(sub_results['total_E'][1] - E_prev[1]) < 1e-6 and
                    abs(sub_results['total_e'] - e_prev) < 1e-6):
                    is_duplicate = True
                    break
        
        # Check optimality: |w - θ| ≤ tolerance
        gap = w - theta
        if abs(gap) <= tolerance or is_duplicate:
            if is_duplicate:
                print(f"\nDuplicate cut detected - stopping algorithm")
            else:
                print(f"\nCONVERGED: |w - θ| = {abs(gap):.10f} ≤ {tolerance}")
            converged = True
        else:
            # Add optimality cut
            cuts.append((sub_results['total_E'], sub_results['total_e']))
            if verbose:
                print(f"\nGAP NOT CLOSED: w - θ = {gap:.6f}")
                cut_terms = [f"{sub_results['total_E'][j]:.4f}*x{j+1}" for j in range(len(sub_results['total_E'])) if abs(sub_results['total_E'][j]) > 1e-10]
                if cut_terms:
                    cut_str = f"Adding cut: {' + '.join(cut_terms)} + θ ≥ {sub_results['total_e']:.4f}"
                else:
                    cut_str = f"Adding cut: θ ≥ {sub_results['total_e']:.4f}"
                print(cut_str)
                print("-" * 80)
    
    # Calculate final objective: first-stage cost + expected second-stage cost
    final_objective = sum(c[i] * x[i] for i in range(len(c))) + sub_results['total_objective']
    
    if verbose:
        print("\n" + "="*80)
        print("L-SHAPED METHOD SUMMARY")
        print("="*80)
        
        if converged:
            print(f"Successfully converged in {iterations} iterations.")
        else:
            print(f"Maximum iterations ({max_iterations}) reached without convergence.")
        
        print(f"\nFinal solution:")
        print(f"  x = [{', '.join([f'{xi:.6f}' for xi in x])}]")
        print(f"  First-stage cost = {sum(c[i] * x[i] for i in range(len(c))):.6f}")
        print(f"  Expected second-stage cost = {sub_results['total_objective']:.6f}")
        print(f"  Total objective = {final_objective:.6f}")
        
        print("\nConvergence history:")
        print("-" * 100)
        print(f"{'Iter':^5} | {'x1':^15} | {'x2':^15} | {'theta':^15} | {'w':^15} | {'gap':^15}")
        print("-" * 100)
        
        for iter_info in history:
            iter_x = iter_info['x']
            print(f"{iter_info['iteration']:^5} | {iter_x[0]:^15.6f} | {iter_x[1]:^15.6f} | {iter_info['theta']:^15.6f} | {iter_info['w']:^15.6f} | {iter_info['gap']:^15.6f}")
    
    return {
        'x': x,
        'theta': theta,
        'objective': final_objective,
        'iterations': iterations,
        'converged': converged,
        'history': history
    }
