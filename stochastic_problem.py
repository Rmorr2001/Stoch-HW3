"""
This module contains the problem data and helper functions for the stochastic programming problem.
"""

def get_problem_data():
    """
    Define the two-stage stochastic problem data.
    
    Returns:
    --------
    dict
        Problem data including scenarios, constraints, and parameters
    """
    # First stage cost coefficients
    c = [100, 150]
    
    # First stage constraints
    A = [[1, 1]]
    b = [120]
    
    # First stage variable bounds
    x_lb = [40, 20]
    
    # Define scenarios
    scenarios = [
        {
            'probability': 0.4,
            'd1': 500,
            'd2': 100,
            'q1': -24,
            'q2': -28,
            'h': [0, 0, 500, 100],  # RHS constants: [constraint1, constraint2, bound_y1, bound_y2]
            'T': [[-60, 0], [0, -80], [0, 0], [0, 0]]  # Technology matrix rows for each constraint
        },
        {
            'probability': 0.6,
            'd1': 300,
            'd2': 300,
            'q1': -28,
            'q2': -32,
            'h': [0, 0, 300, 300],  # RHS constants: [constraint1, constraint2, bound_y1, bound_y2]
            'T': [[-60, 0], [0, -80], [0, 0], [0, 0]]  # Technology matrix rows for each constraint
        }
    ]
    
    return {
        'c': c,
        'A': A,
        'b': b,
        'x_lb': x_lb,
        'scenarios': scenarios
    }

def calculate_rhs(h, T, x):
    """
    Calculate h - Tx for the right-hand side of the subproblem constraints.
    
    Parameters:
    -----------
    h : list
        RHS constants
    T : list of lists
        Technology matrix
    x : list
        First stage decisions
    
    Returns:
    --------
    list
        h - Tx values
    """
    result = []
    for i in range(len(h)):
        value = h[i]
        for j in range(len(x)):
            value -= T[i][j] * x[j]
        result.append(value)
    return result

def calculate_optimality_components(pi, h, T, probability):
    """
    Calculate e and E components for optimality cuts.
    
    Parameters:
    -----------
    pi : list
        Simplex multipliers (dual values)
    h : list
        RHS constants
    T : list of lists
        Technology matrix
    probability : float
        Scenario probability
    
    Returns:
    --------
    tuple
        (e, E) values for the cut
    """
    # Make sure pi has the correct length
    if len(pi) != len(h):
        pi_adjusted = pi + [0] * (len(h) - len(pi))
    else:
        pi_adjusted = pi
    
    # Calculate e = π^T · h
    e = 0
    for i in range(len(h)):
        e += pi_adjusted[i] * h[i]
    e *= probability
    
    # Calculate E = π^T · T
    E = [0] * len(T[0])
    for j in range(len(E)):
        for i in range(len(T)):
            if i < len(pi_adjusted):  # Safety check
                E[j] += pi_adjusted[i] * T[i][j]
        E[j] *= probability
    
    return e, E