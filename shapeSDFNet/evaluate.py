import numpy as np
import matplotlib.pyplot as plt
import datetime

def evaluate_model(true_func, model_func, x_start=0, x_end=1, num_points=100, plot=True):
    
    # Evaluate the accuracy of a modeled function compared to a true function.
    # 
    # Both functions take as input a NumPy array of shape (3,) and return a scalar output.
    # 
    # 
    # true_func: The true function. Expects input: NumPy array of shape (3,).
    # model_func: The modeled function. Expects input: NumPy array of shape (3,).
    # x_start: Lower bound for generating each element of the input array.
    # x_end: Upper bound for generating each element of the input array.
    # num_points: Number of random input samples to generate.
    # plot: If True, creates and saves a scatter plot comparing the outputs.
    # save_filename: The filename to save the plot to.
    #    
    # - 'MAE': Mean Absolute Error.
    # - 'MSE': Mean Squared Error.
    # - 'RMSE': Root Mean Squared Error.
    # - 'Max Error': Maximum absolute error.
    # - 'R2': Coefficient of determination.
    # - 'MAPE': Mean Absolute Percentage Error (if applicable, else None).
    
    save_filename = f"img/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    
    # Generate random test input points (each is a NumPy array of shape (3,))
    x_values = np.random.uniform(x_start, x_end, (num_points, 3))
    
    # Evaluate both functions for each input.
    true_y = np.array([true_func(x) for x in x_values])
    model_y = np.array([model_func(x) for x in x_values])
    
    # Calculate error metrics.
    mae = np.mean(np.abs(true_y - model_y))
    mse = np.mean((true_y - model_y) ** 2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(true_y - model_y))
    
    # Compute R-squared (coefficient of determination).
    ss_tot = np.sum((true_y - np.mean(true_y)) ** 2)
    ss_res = np.sum((true_y - model_y) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    
    # Compute Mean Absolute Percentage Error (MAPE) if possible.
    if np.all(true_y != 0):
        mape = np.mean(np.abs((true_y - model_y) / true_y)) * 100
    else:
        mape = None
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Max Error': max_error,
        'R2': r2,
        'MAPE': mape
    }
    
    # Print evaluation metrics.
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Create and save the scatter plot.
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(true_y, model_y, alpha=0.6, label="Data points")
        # Plot the perfect fit line (y = x).
        min_val = min(true_y.min(), model_y.min())
        max_val = max(true_y.max(), model_y.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit (y=x)")
        plt.xlabel("True Function Output")
        plt.ylabel("Modeled Function Output")
        plt.title("Comparison of True vs Modeled Outputs")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_filename)  # Save the plot to a file.
        plt.close()  # Close the figure to free up memory.
    
    return metrics