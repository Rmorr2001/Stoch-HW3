"""
Main program to run the L-shaped method algorithm with detailed output.
"""
from l_shaped_method import run_l_shaped_method

def main():
    """
    Run the L-shaped method with detailed output and display results.
    """
    # Set a larger number of iterations and smaller tolerance
    #For this assignment the known amount of cuts is 5 SO if it's at 10 that's bad
    # Verbose = True which means it will print out every stage of the process.
    result = run_l_shaped_method(max_iterations=10, tolerance=1e-8, verbose=True)
    
    # Print final summary
    print("\nFINAL SOLUTION SUMMARY:")
    print("=" * 80)
    print(f"x = [{', '.join([f'{xi:.6f}' for xi in result['x']])}]")
    print(f"Objective value = {result['objective']:.6f}")
    print(f"Number of iterations = {result['iterations']}")
    print(f"Convergence status = {'Converged' if result['converged'] else 'Not converged'}")

if __name__ == "__main__":
    main()
