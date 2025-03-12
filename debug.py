"""
Debug utilities to help diagnose issues with the L-shaped method implementation.
"""
import pyomo.environ as pyo
from stochastic_problem import get_problem_data, calculate_rhs

def test_master_problem_manually():
    """
    Test the master problem formulation manually to verify it works as expected.
    """
    # Get problem data
    data = get_problem_data()
    c = data['c']
    A = data['A']
    b = data['b']
    x_lb = data['x_lb']
    
    # Create a concrete model
    model = pyo.ConcreteModel(name="Master_Problem_Test")
    
    # Define indices
    model.n = range(len(c))
    model.m = range(len(A))
    
    # Define variables
    model.x = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    
    # Set lower bounds
    for i in model.n:
        model.x[i].setlb(x_lb[i])
    
    # Define objective function
    model.obj = pyo.Objective(
        expr=sum(c[i] * model.x[i] for i in model.n),
        sense=pyo.minimize
    )
    
    # Define first-stage constraints
    model.constraints = pyo.ConstraintList()
    for i in model.m:
        model.constraints.add(
            sum(A[i][j] * model.x[j] for j in model.n) <= b[i]
        )
    
    # Solve the model
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=True)
    
    # Print results
    print("Solver status:", results.solver.status)
    print("Termination condition:", results.solver.termination_condition)
    
    if results.solver.status == pyo.SolverStatus.ok:
        print("x values:")
        for i in model.n:
            print(f"x[{i}] = {model.x[i].value}")
        print("Objective value:", model.obj())
    
    return model

def test_subproblem_manually():
    """
    Test a subproblem formulation manually to verify it works as expected.
    """
    # Get problem data
    data = get_problem_data()
    scenarios = data['scenarios']
    
    # Use first scenario and initial values [40, 20] for x
    scenario = scenarios[0]
    x = [40, 20]
    
    # Calculate RHS
    h = scenario['h']
    T = scenario['T']
    rhs = calculate_rhs(h, T, x)
    
    # Extract other parameters
    q1 = scenario['q1']
    q2 = scenario['q2']
    d1 = scenario['d1']
    d2 = scenario['d2']
    
    # Create a concrete model
    model = pyo.ConcreteModel(name="Subproblem_Test")
    
    # Define variables
    model.y = pyo.Var([0, 1], domain=pyo.NonNegativeReals)
    
    # Set upper bounds
    model.y[0].setub(d1)
    model.y[1].setub(d2)
    
    # Define objective function
    model.obj = pyo.Objective(
        expr=q1 * model.y[0] + q2 * model.y[1],
        sense=pyo.minimize
    )
    
    # Define constraints
    model.constraints = pyo.ConstraintList()
    
    # 6y1 + 10y2 ≤ 60x1
    model.constraints.add(6 * model.y[0] + 10 * model.y[1] <= rhs[0])
    
    # 8y1 + 5y2 ≤ 80x2
    model.constraints.add(8 * model.y[0] + 5 * model.y[1] <= rhs[1])
    
    # Solve the model
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=True)
    
    # Print results
    print("Solver status:", results.solver.status)
    print("Termination condition:", results.solver.termination_condition)
    
    if results.solver.status == pyo.SolverStatus.ok:
        print("y values:")
        for i in [0, 1]:
            print(f"y[{i}] = {model.y[i].value}")
        print("Objective value:", model.obj())
        
        # Add dual information
        print("Constraint duals:")
        for i, c in enumerate(model.constraints):
            print(f"Constraint {i}: {model.dual[c]}")
    
    return model

if __name__ == "__main__":
    print("Testing master problem...")
    print("-" * 60)
    test_master_problem_manually()
    
    print("\nTesting subproblem...")
    print("-" * 60)
    test_subproblem_manually()