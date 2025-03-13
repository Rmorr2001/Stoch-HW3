"""
Analysis of convergence issues in the L-shaped method implementation.
"""
import numpy as np
from stochastic_problem import get_problem_data, calculate_rhs

def analyze_problem_structure():
    """
    Analyze the structure of the problem to understand convergence behavior.
    """
    # Get problem data
    data = get_problem_data()
    c = data['c']
    scenarios = data['scenarios']
    
    print("="*80)
    print("PROBLEM STRUCTURE ANALYSIS")
    print("="*80)
    
    print("\nFirst stage coefficients:")
    print(f"c = {c}")
    
    print("\nScenario information:")
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1} (p = {scenario['probability']}):")
        print(f"  q = [{scenario['q1']}, {scenario['q2']}]")
        print(f"  d = [{scenario['d1']}, {scenario['d2']}]")
        print(f"  h = {scenario['h']}")
        print(f"  T = {scenario['T']}")
    
    # Check the problem against the optimal solution (from slides)
    test_points = [
        ([40, 20], "Initial solution"),
        ([40, 80], "Second iteration"),
        ([46.67, 36.25], "Expected optimal solution"),
        ([75.42, 44.58], "Current solution")
    ]
    
    print("\nEvaluating key points:")
    print("-"*80)
    print(f"{'Point':^15} | {'Description':^20} | {'First Stage Cost':^17} | {'Second Stage':^13} | {'Total':^10}")
    print("-"*80)
    
    for point, desc in test_points:
        # Compute first stage cost
        first_stage = sum(c[i] * point[i] for i in range(len(c)))
        
        # Compute second stage cost for each scenario
        second_stage = 0
        for scenario in scenarios:
            # Calculate RHS
            h = scenario['h']
            T = scenario['T']
            rhs = calculate_rhs(h, T, point)
            
            # Solve second stage problem analytically for this specific problem
            # Assume we know the optimal solution structure
            q1, q2 = scenario['q1'], scenario['q2']
            d1, d2 = scenario['d1'], scenario['d2']
            
            # For this problem, we typically want to maximize both y1 and y2 since q1 and q2 are negative
            # Subject to the constraints:
            # 6y1 + 10y2 ≤ 60x1
            # 8y1 + 5y2 ≤ 80x2
            # 0 ≤ y1 ≤ d1
            # 0 ≤ y2 ≤ d2
 
            # Try to maximize y1 and y2 based on the stronger constraint
            max_y1 = min(d1, rhs[0]/6, rhs[1]/8)
            max_y2 = min(d2, (rhs[0] - 6*max_y1)/10, (rhs[1] - 8*max_y1)/5)
            
            # Compute objective value
            obj = q1 * max_y1 + q2 * max_y2
            second_stage += scenario['probability'] * obj
        
        # Compute total cost
        total = first_stage + second_stage
        
        print(f"{str(point):^15} | {desc:^20} | {first_stage:^17.4f} | {second_stage:^13.4f} | {total:^10.4f}")
    
    print("-"*80)
    print("\nNote: This analysis uses an approximation for the second stage solution.")
    print("The actual optimal solution should be computed using a proper LP solver.")

if __name__ == "__main__":
    analyze_problem_structure()
