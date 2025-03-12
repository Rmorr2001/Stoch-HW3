"""
This module implements the subproblem solvers for the L-shaped method.
"""
from SimplexMultipliers import solve_lp_with_duals
from stochastic_problem import calculate_rhs, calculate_optimality_components
import numpy as np

def transform_dual_vector(raw_pi, num_constraints, num_variables, y_values, upper_bounds):
    """
    Transform the raw dual vector to the format expected by the L-shaped method.
    Generic implementation without any special case handling.
    
    Parameters:
    -----------
    raw_pi : list
        Raw dual vector from the solver
    num_constraints : int
        Number of constraints
    num_variables : int
        Number of variables
    y_values : list
        Solution values for variables
    upper_bounds : list
        Upper bounds for variables
    """
    # Initialize result vector
    result = [0] * (num_constraints + num_variables)
    
    # Copy constraint duals directly
    for i in range(min(num_constraints, len(raw_pi))):
        result[i] = raw_pi[i]
    
    # Process variable bounds - generic approach
    # In the raw vector, upper bound duals start at position num_constraints + num_variables
    ub_start = num_constraints + num_variables
    
    # For each variable, check if it's at upper bound
    for i in range(num_variables):
        if abs(y_values[i] - upper_bounds[i]) < 1e-6:
            # Variable is at upper bound, look for a non-zero dual in the raw vector
            for j in range(num_constraints, len(raw_pi)):
                if abs(raw_pi[j]) > 1e-6:
                    # Found a non-zero dual value, assume it corresponds to this variable
                    # In L-shaped method format, the upper bound dual has negative sign
                    result[num_constraints + i] = -abs(raw_pi[j])
                    # Remove this value so it's not used again
                    raw_pi[j] = 0
                    break
    
    return result

def solve_subproblem(x, scenario, iteration_num):
    """
    Solve a second-stage subproblem for given first-stage decisions and scenario parameters.
    
    Parameters:
    -----------
    x : list
        First stage decision values
    scenario : dict
        Scenario parameters
    iteration_num : int
        Current iteration number
    
    Returns:
    --------
    dict
        Contains solution information including objective and dual values
    """
    # Extract scenario data
    d1 = scenario['d1']
    d2 = scenario['d2']
    q1 = scenario['q1']
    q2 = scenario['q2']
    h = scenario['h']
    T = scenario['T']
    probability = scenario['probability']
    
    # Calculate right-hand side: h - Tx
    rhs = calculate_rhs(h, T, x)
    
    # Setup the constraint matrix (W)
    W = [
        [6, 10],  # 6y1 + 10y2 ≤ 60x1
        [8, 5]    # 8y1 + 5y2 ≤ 80x2
    ]
    
    # Setup objective coefficients
    objective_coeffs = [q1, q2]
    
    # Variable bounds
    var_lower_bounds = [0, 0]
    var_upper_bounds = [d1, d2]
    
    # Solve the LP using the provided function
    result = solve_lp_with_duals(
        objective_coeffs=objective_coeffs,
        constraint_matrix=W,
        constraint_rhs=rhs[:2],  # Only the first two elements are actual constraints
        var_lower_bounds=var_lower_bounds,
        var_upper_bounds=var_upper_bounds,
        problem_name=f"Subproblem_p{probability}"
    )
    
    if result['status'] != 'optimal':
        raise ValueError(f"Subproblem could not be solved optimally: {result['status']}")
    
    # Get solution values
    y_values = result['variable_values']
    objective_value = result['objective_value']
    
    # Get raw pi vector from solver result
    raw_pi_vector = result['pi_vector']
    
    # Transform the raw pi vector to the expected format
    pi_vector = transform_dual_vector(
        raw_pi_vector,
        num_constraints=2,
        num_variables=2,
        y_values=y_values,
        upper_bounds=var_upper_bounds
    )
    
    # Calculate components for the optimality cut
    e, E = calculate_optimality_components(pi_vector, h, T, probability)
    
    # Calculate w = e - E*x
    w = e
    for i in range(len(x)):
        w -= E[i] * x[i]
    
    return {
        'objective_value': objective_value,
        'y_values': y_values,
        'pi_vector': pi_vector,
        'e': e,
        'E': E,
        'w': w
    }

def solve_all_subproblems(x, scenarios, iteration_num):
    """
    Solve all scenario subproblems and aggregate the results.
    
    Parameters:
    -----------
    x : list
        First stage decision values
    scenarios : list
        List of scenario data
    iteration_num : int
        Current iteration number
    
    Returns:
    --------
    dict
        Contains aggregated results
    """
    total_e = 0
    total_E = [0] * len(x)
    total_objective = 0
    
    results = []
    
    for scenario in scenarios:
        result = solve_subproblem(x, scenario, iteration_num)
        results.append(result)
        
        total_e += result['e']
        for i in range(len(x)):
            total_E[i] += result['E'][i]
        
        total_objective += scenario['probability'] * result['objective_value']
    
    # Calculate w = e - Ex
    total_w = total_e
    for i in range(len(x)):
        total_w -= total_E[i] * x[i]
    
    return {
        'scenario_results': results,
        'total_objective': total_objective,
        'total_e': total_e,
        'total_E': total_E,
        'total_w': total_w
    }