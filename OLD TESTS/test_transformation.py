"""
Test the algorithmic dual transformation for the L-shaped method.
"""
from SimplexMultipliers import solve_lp_with_duals
from subproblems import transform_dual_vector

def test_dual_transformation():
    """
    Test the algorithmic transformation of dual vectors.
    """
    print("Testing Dual Vector Transformation")
    print("-" * 60)
    
    # Example Problem 1 (Scenario 1)
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
    
    raw_pi1 = result1['pi_vector']
    transformed_pi1 = transform_dual_vector(raw_pi1, 2, 2)
    
    # Expected value from the slides
    expected_pi1 = [0, -3, 0, -13]
    
    print(f"Scenario 1 (p=0.4), Iteration 1:")
    print(f"  Raw pi: {raw_pi1}")
    print(f"  Transformed pi: {transformed_pi1}")
    print(f"  Expected pi: {expected_pi1}")
    print(f"  Match: {transformed_pi1 == expected_pi1}")
    
    # Example Problem 2 (Scenario 2)
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
    
    raw_pi2 = result2['pi_vector']
    transformed_pi2 = transform_dual_vector(raw_pi2, 2, 2)
    
    # Expected value from the slides
    expected_pi2 = [-2.32, -1.76, 0, 0]
    
    print(f"\nScenario 2 (p=0.6), Iteration 1:")
    print(f"  Raw pi: {raw_pi2}")
    print(f"  Transformed pi: {transformed_pi2}")
    print(f"  Expected pi: {expected_pi2}")
    print(f"  Close match: {all(abs(a-b) < 0.1 for a, b in zip(transformed_pi2, expected_pi2))}")
    
    # Test the next iteration scenarios
    # This simulates the Scenario 1, Iteration 2 case with different x values
    result3 = solve_lp_with_duals(
        objective_coeffs=[-24, -28],
        constraint_matrix=[
            [6, 10],
            [8, 5]
        ],
        constraint_rhs=[2400, 6400],  # Modify RHS to simulate x=[40, 80]
        var_lower_bounds=[0, 0],
        var_upper_bounds=[500, 100],
        problem_name="Example_3"
    )
    
    raw_pi3 = result3['pi_vector']
    transformed_pi3 = transform_dual_vector(raw_pi3, 2, 2)
    
    # Expected value for Scenario 1, Iteration 2
    expected_pi3 = [-4, 0, 0, 0]
    
    print(f"\nScenario 1 (p=0.4), Iteration 2:")
    print(f"  Raw pi: {raw_pi3}")
    print(f"  Transformed pi: {transformed_pi3}")
    print(f"  Expected pi: {expected_pi3}")
    print(f"  Close match: {all(abs(a-b) < 0.1 for a, b in zip(transformed_pi3, expected_pi3))}")

if __name__ == "__main__":
    test_dual_transformation()