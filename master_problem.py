"""
This module implements the master problem for the L-shaped method.
"""
import pyomo.environ as pyo

def solve_master_problem(c, A, b, x_lb, cuts=None):
    """
    Solve the master problem with current optimality cuts.
    
    Parameters:
    -----------
    c : list
        First stage cost coefficients
    A : list of lists
        First stage constraint matrix
    b : list
        First stage RHS
    x_lb : list
        Lower bounds for first stage variables
    cuts : list
        List of optimality cuts, each represented as a tuple (E, e)
    
    Returns:
    --------
    dict
        Contains the optimal solution and objective value
    """
    # Create a concrete model
    model = pyo.ConcreteModel(name="Master_Problem")
    
    # Define indices
    model.n = range(len(c))
    model.m = range(len(A))
    
    # Define variables
    model.x = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    
    # Set lower bounds
    for i in model.n:
        model.x[i].setlb(x_lb[i])
    
    # Only include theta if there are cuts
    if cuts and len(cuts) > 0:
        model.theta = pyo.Var(domain=pyo.Reals)
        # Define objective with theta
        model.obj = pyo.Objective(
            expr=sum(c[i] * model.x[i] for i in model.n) + model.theta,
            sense=pyo.minimize
        )
    else:
        # Without cuts, no need for theta
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
    
    # Add optimality cuts
    if cuts and len(cuts) > 0:
        model.cuts = pyo.ConstraintList()
        for E, e in cuts:
            model.cuts.add(
                sum(E[j] * model.x[j] for j in model.n) + model.theta >= e
            )
    
    # Solve the model
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=False)
    
    # Check if the model was solved successfully
    if results.solver.status != pyo.SolverStatus.ok or \
       results.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"Warning: Solver status: {results.solver.status}, termination condition: {results.solver.termination_condition}")
        return {
            'x': [x_lb[i] for i in range(len(c))],
            'theta': float('-inf'),
            'objective': sum(c[i] * x_lb[i] for i in range(len(c))),
            'status': str(results.solver.termination_condition)
        }
    
    # Extract solution information
    try:
        x_values = [model.x[i].value for i in model.n]
        
        # Get theta value if it exists
        if cuts and len(cuts) > 0:
            theta_value = model.theta.value
        else:
            theta_value = float('-inf')
        
        # Calculate objective
        objective_value = sum(c[i] * x_values[i] for i in range(len(c)))
        if cuts and len(cuts) > 0:
            objective_value += theta_value
            
    except Exception as e:
        # Fallback in case of errors
        print(f"Warning: Could not retrieve solution values properly. Error: {e}")
        print("Using fallback method.")
        x_values = [x_lb[i] for i in range(len(c))]
        theta_value = float('-inf')
        objective_value = sum(c[i] * x_lb[i] for i in range(len(c)))
    
    return {
        'x': x_values,
        'theta': theta_value,
        'objective': objective_value,
        'status': str(results.solver.termination_condition)
    }