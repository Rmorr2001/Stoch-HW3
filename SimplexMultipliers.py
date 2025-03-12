import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def solve_lp_with_duals(objective_coeffs, constraint_matrix, constraint_rhs, 
                        var_lower_bounds, var_upper_bounds, problem_name="LP Problem"):
    """
    Solve a linear programming problem and return the simplex multipliers.
    
    Parameters:
    -----------
    objective_coeffs : list
        Coefficients of the objective function (minimization)
    constraint_matrix : list of lists
        Matrix representing the constraints (each row is a constraint)
    constraint_rhs : list
        Right-hand side values of the constraints
    var_lower_bounds : list
        Lower bounds for each variable
    var_upper_bounds : list
        Upper bounds for each variable
    problem_name : str, optional
        Name of the problem for display purposes
        
    Returns:
    --------
    dict
        Contains objective value, variable values, and simplex multipliers
    """
    # Create a concrete model
    model = pyo.ConcreteModel(name=problem_name)
    
    # Set up the indices
    num_vars = len(objective_coeffs)
    num_constraints = len(constraint_rhs)
    model.I = pyo.RangeSet(0, num_vars-1)
    model.J = pyo.RangeSet(0, num_constraints-1)
    
    # Define variables
    model.y = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    
    # Set variable bounds
    for i in model.I:
        model.y[i].setlb(var_lower_bounds[i])
        model.y[i].setub(var_upper_bounds[i])
    
    # Define objective function (minimization)
    model.obj = pyo.Objective(
        expr=sum(objective_coeffs[i] * model.y[i] for i in model.I),
        sense=pyo.minimize
    )
    
    # Define constraints
    model.constraints = pyo.ConstraintList()
    for j in model.J:
        model.constraints.add(
            sum(constraint_matrix[j][i] * model.y[i] for i in model.I) <= constraint_rhs[j]
        )
    
    # Define suffixes for retrieving dual values
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)  # Reduced costs
    
    # Solve the model
    solver = SolverFactory('glpk')
    results = solver.solve(model)
    
    # Check if solution is optimal
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        # Extract primal solution
        obj_value = pyo.value(model.obj)
        y_values = [pyo.value(model.y[i]) for i in model.I]
        
        # Create raw dual vector that includes all information
        raw_pi_vector = []
        
        # Extract constraint duals
        for j in range(len(model.constraints)):
            constraint = model.constraints[j+1]  # Pyomo uses 1-based indexing for ConstraintList
            dual = model.dual.get(constraint, 0.0)
            raw_pi_vector.append(dual)
        
        # Extract variable bound information (for debugging)
        for i in model.I:
            # Lower bound duals
            raw_pi_vector.append(0)  # Placeholder for simplicity
            
            # Upper bound duals
            if abs(pyo.value(model.y[i]) - var_upper_bounds[i]) < 1e-6:
                reduced_cost = model.rc.get(model.y[i], 0.0)
                # For upper bounds, if the reduced cost is negative, the bound is binding
                if reduced_cost < 0:
                    raw_pi_vector.append(abs(reduced_cost))  # Store as positive value
                else:
                    raw_pi_vector.append(0)
            else:
                raw_pi_vector.append(0)
        
        # Return results
        return {
            'status': 'optimal',
            'objective_value': obj_value,
            'variable_values': y_values,
            'pi_vector': raw_pi_vector,
            'raw_output': True  # Flag indicating this is raw output for debugging
        }
    else:
        return {
            'status': 'failed',
            'termination_condition': str(results.solver.termination_condition)
        }

def format_pi_vector(pi_vector):
    """Format the pi vector to match the example format, filtering out zeros."""
    non_zero = []
    for value in pi_vector:
        if abs(value) > 1e-6:  # Filter out near-zero values
            non_zero.append(round(value, 2))
    return tuple(non_zero)

def solve_examples():
    """Solve the example problems provided in the images."""
    # Example Problem 1
    print("Solving Example Problem 1")
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
    
    if result1['status'] == 'optimal':
        print(f"Status: {result1['status']}")
        print(f"Objective value: {result1['objective_value']}")
        print(f"Variable values: {result1['variable_values']}")
        print(f"Raw pi vector: {result1['pi_vector']}")
        
        # Expected values from slides
        expected_pi = [0, -3, 0, -13]
        print(f"Expected pi = {expected_pi}")
        print(f"Actual pi = {result1['pi_vector']}")
    else:
        print(f"Example Problem 1 failed: {result1['termination_condition']}")
    
    # Example Problem 2
    print("\nSolving Example Problem 2")
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
    
    if result2['status'] == 'optimal':
        print(f"Status: {result2['status']}")
        print(f"Objective value: {result2['objective_value']}")
        print(f"Variable values: {result2['variable_values']}")
        print(f"Raw pi vector: {result2['pi_vector']}")
        
        # Expected values from slides
        expected_pi = [-2.32, -1.76, 0, 0]
        print(f"Expected pi = {expected_pi}")
        print(f"Actual pi = {result2['pi_vector']}")
    else:
        print(f"Example Problem 2 failed: {result2['termination_condition']}")


if __name__ == "__main__":
    # Solve the examples from the images
    solve_examples()