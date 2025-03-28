import matplotlib.pyplot as plt
import datetime

def plot_point_cloud(point_list, label_list, label):
    # Generate random data
    x = []; y = []; z = []
    
    for i in range(len(label_list)):
        if label_list[i] == label:
            x.append(point_list[i][0])
            y.append(point_list[i][1])
            z.append(point_list[i][2])

    # Set up a figure with a black background
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Create scatter plot with white points
    ax.scatter(x, y, z, c='white')

    # Optional: remove axes lines and labels for a cleaner look
    ax.set_axis_off()
    
    save_filename = f"img/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"

    # Save the figure
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='black')

    # Close the figure to free memory
    plt.close()