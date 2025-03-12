# L-Shaped Method Implementation

This project implements the L-shaped method for solving two-stage stochastic linear programming problems. The implementation focuses on solving the following problem:

```
min 100x1 + 150x2 + Eξ(q1y1 + q2y2)
s.t. x1 + x2 ≤ 120
     6y1 + 10y2 ≤ 60x1
     8y1 + 5y2 ≤ 80x2
     y1 ≤ d1
     y2 ≤ d2
     x1 ≥ 40
     x2 ≥ 20
     y1, y2 ≥ 0
```

where ξ = (d1, d2, q1, q2) takes the values:
- (500, 100, -24, -28), with probability 0.4 (Scenario 1)
- (300, 300, -28, -32), with probability 0.6 (Scenario 2)

## Algorithm Overview

The L-shaped method is a decomposition algorithm for solving two-stage stochastic linear programs. It works by:

1. Solving a master problem to get first-stage decisions
2. Solving scenario subproblems to evaluate recourse costs and obtain dual information
3. Generating optimality cuts based on dual information
4. Adding these cuts to the master problem and iterating until convergence

## Project Structure

This repository contains several Python files that work together:

1. main.py: Entry point for running the algorithm
2. l_shaped_method.py: Implements the main L-shaped method algorithm
3. master_problem.py: Handles the master problem and optimality cuts
4. subproblems.py: Solves the second-stage recourse problems and processes dual values
5. stochastic_problem.py: Contains problem data and helper functions
6. SimplexMultipliers.py: Solves LPs and extracts dual values
7. workflow.py: Provides detailed explanation of the algorithm, assumptions, and implementation

## Implementation Details

### Dual Value Processing

The core challenge in this implementation is correctly processing the dual values (simplex multipliers) from the LP solver. This approach:

1. Extract raw dual information from the solver
2. Transform this information into the format required by the L-shaped method
3. Calculate optimality cut components using the transformed dual values

The dual transformation function in `subproblems.py` handles this process generically without special case handling.

### Optimality Cuts

Optimality cuts have the form:
```
Ex + θ ≥ e
```

Where:
- E is derived from dual values and the technology matrix
- e is derived from dual values and right-hand side constants
- θ is a variable representing the expected recourse cost

### Convergence

The algorithm converges when the gap between θ (the approximation of the recourse cost) and w (the actual recourse cost) is sufficiently small.

## How to Run

To run the implementation:

```bash
python main.py
```

This will execute the L-shaped method algorithm with detailed output showing each iteration, including:
- The master problem formulation and solution
- The subproblem formulations and solutions for each scenario
- The dual values and cut components
- The convergence progress
