"""
Test the SimplexMultipliers module with the example problems from the workflow.
"""
from SimplexMultipliers import solve_lp_with_duals
from subproblems import convert_to_expected_format

def test_example_problems():
    """
    Test the simplex multipliers function with the two example problems.
    """
    print("Testing Example Problem 1:")
    print("-" * 60)
    
    # Example Problem 1 (from workflow.py)
    # min -24y1 - 28y2
    # s.t. 6y1 + 10y2 ≤ 2400
    #      8y1 + 5y2 ≤ 1600
    #      0 ≤ y1 ≤ 500
    #      0 ≤ y2 ≤ 100
    result1 = solve_lp_with_duals(
        objective_coeffs=[-24, -28],
        constraint_matrix=[
            [6, 10],
            [8, 5]
        ],
        constraint_rhs=[2400, 1600],
        var_lower_bounds=[0, 0],
        var_upper_bounds=[500, 100],
        problem_name="Example_1"
    )
    
    print(f"Status: {result1['status']}")
    print(f"Objective value: {result1['objective_value']}")
    print(f"Variable values: {result1['variable_values']}")
    print(f"Raw pi vector: {result1['pi_vector']}")
    
    # Convert to expected format
    expected_pi = [0, -3, 0, -13]
    converted_pi = convert_to_expected_format(result1['pi_vector'], 0.4, 1)
    print(f"Expected pi = {expected_pi}")
    print(f"Converted pi = {converted_pi}")
    
    print("\nTesting Example Problem 2:")
    print("-" * 60)
    
    # Example Problem 2
    # min -28y1 - 32y2
    # s.t. 6y1 + 10y2 ≤ 2400
    #      8y1 + 5y2 ≤ 1600
    #      0 ≤ y1 ≤ 300
    #      0 ≤ y2 ≤ 300
    result2 = solve_lp_with_duals(
        objective_coeffs=[-28, -32],
        constraint_matrix=[
            [6, 10],
            [8, 5]
        ],
        constraint_rhs=[2400, 1600],
        var_lower_bounds=[0, 0],
        var_upper_bounds=[300, 300],
        problem_name="Example_2"
    )
    
    print(f"Status: {result2['status']}")
    print(f"Objective value: {result2['objective_value']}")
    print(f"Variable values: {result2['variable_values']}")
    print(f"Raw pi vector: {result2['pi_vector']}")
    
    # Convert to expected format
    expected_pi2 = [-2.32, -1.76, 0, 0]
    converted_pi2 = convert_to_expected_format(result2['pi_vector'], 0.6, 1)
    print(f"Expected pi = {expected_pi2}")
    print(f"Converted pi = {converted_pi2}")

if __name__ == "__main__":
    test_example_problems()