import matplotlib.pyplot as plt

# Function to read the file and extract abs_divergence and sq_divergence lists
def read_data(file_name):
    abs_divergence = []
    sq_divergence = []
    Ld =[]
    Lp =[]
    
    with open(file_name, 'r') as file:
        # Skip the first line (the headers)
        next(file)
        
        # Read each line of data
        for line in file:
            data = line.strip().split()  # Split the line into two parts (abs_divergence and sq_divergence)
            abs_divergence.append(float(data[0]))  # Add the first value (abs_divergence)
            sq_divergence.append(float(data[1]))  # Add the second value (sq_divergence)
            Ld.append(float(data[2]))  # Add the second value (sq_divergence)
            Lp.append(float(data[3]))
    
    return abs_divergence, sq_divergence, Ld, Lp

# Function to plot abs_divergence and sq_divergence
def plot_data(abs_divergence, sq_divergence, Ld, Lp):
    iterations = list(range(len(abs_divergence)))  # Create a list of iterations
    
    plt.plot(iterations, abs_divergence, label='Absolute Divergence')  # Plot abs_divergence
    plt.plot(iterations, sq_divergence, label='Square Divergence')  # Plot sq_divergence
    plt.plot(iterations, Ld, label='Ld')  # Plot Ld
    plt.plot(iterations, Lp, label='Lp')  # Plot Lp
    
    plt.xlabel('Iterations')  # X-axis label
    plt.ylabel('Value')  # Y-axis label
    # plt.title('Divergences, Ld, and Lp Evolution over Iterations')  # Plot title
    plt.legend()  # Display a legend to identify the lines
    plt.grid(True)  # Add grid lines for better readability
    
    plt.show()  # Show the plot

# File where the data is stored
file_name = 'log_output_cube_benckmark_0.02_my_pruned_12FPS.txt'

# Read the data from the file
abs_divergence, sq_divergence, Ld, Lp = read_data(file_name)

# Plot the data
plot_data(abs_divergence, sq_divergence, Ld, Lp)
