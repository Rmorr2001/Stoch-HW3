"""
Verification of the simplex multiplier transformation for Scenario 2, Iteration 2.
"""
from SimplexMultipliers import solve_lp_with_duals
from subproblems import transform_dual_vector
from stochastic_problem import calculate_rhs, calculate_optimality_components

def verify_scenario2_iteration2():
    """
    Verify the simplex multiplier transformation for Scenario 2, Iteration 2.
    """
    print("Verifying Scenario 2, Iteration 2 Transformation")
    print("=" * 80)
    
    # First-stage decision from Iteration 2
    x = [40, 80]
    
    # Scenario 2 data
    probability = 0.6
    d1 = 300
    d2 = 300
    q1 = -28
    q2 = -32
    h = [0, 0, 300, 300]
    T = [[-60, 0], [0, -80], [0, 0], [0, 0]]
    
    # Calculate RHS values
    rhs = calculate_rhs(h, T, x)
    print(f"RHS values: {rhs}")
    
    # Solve the LP
    result = solve_lp_with_duals(
        objective_coeffs=[q1, q2],
        constraint_matrix=[
            [6, 10],
            [8, 5]
        ],
        constraint_rhs=rhs[:2],
        var_lower_bounds=[0, 0],
        var_upper_bounds=[d1, d2],
        problem_name="Scenario2_Iter2"
    )
    
    print("\nLP Solution:")
    print(f"Status: {result['status']}")
    print(f"Variable values: y = {result['variable_values']}")
    print(f"Objective value: {result['objective_value']}")
    print(f"Raw pi vector: {result['pi_vector']}")
    
    # Transform the raw pi vector
    transformed_pi = transform_dual_vector(
        result['pi_vector'],
        num_constraints=2,
        num_variables=2,
        y_values=result['variable_values'],
        upper_bounds=[d1, d2]
    )
    
    print(f"Transformed pi vector: {transformed_pi}")
    
    # Expected value from slides
    expected_pi = [-3.2, 0.0, -8.8, 0]
    print(f"Expected pi vector: {expected_pi}")
    print(f"Match with expected: {all(abs(a-b) < 0.1 for a, b in zip(transformed_pi, expected_pi))}")
    
    # Calculate cut components
    e, E = calculate_optimality_components(transformed_pi, h, T, probability)
    w = e - E[0] * x[0] - E[1] * x[1]
    
    print("\nCut Components (Using Transformed Pi):")
    print(f"e = {e}")
    print(f"E = {E}")
    print(f"w = {w}")
    
    # Expected values from slides
    expected_e = 1584.0
    expected_E = [211.2, 0.0]
    expected_w = expected_e - expected_E[0] * x[0] - expected_E[1] * x[1]
    
    print("\nExpected Cut Components:")
    print(f"e = {expected_e}")
    print(f"E = {expected_E}")
    print(f"w = {expected_w}")
    
    # Now using the corrected values
    corrected_pi = [-3.2, 0.0, -8.8, 0]
    e_corr, E_corr = calculate_optimality_components(corrected_pi, h, T, probability)
    w_corr = e_corr - E_corr[0] * x[0] - E_corr[1] * x[1]
    
    print("\nCut Components (Using Corrected Pi):")
    print(f"e = {e_corr}")
    print(f"E = {E_corr}")
    print(f"w = {w_corr}")
    
    return {
        'raw_pi': result['pi_vector'],
        'transformed_pi': transformed_pi,
        'expected_pi': expected_pi,
        'e': e,
        'E': E,
        'w': w,
        'expected_e': expected_e,
        'expected_E': expected_E,
        'expected_w': expected_w,
        'e_corr': e_corr,
        'E_corr': E_corr,
        'w_corr': w_corr
    }

if __name__ == "__main__":
    verify_scenario2_iteration2()