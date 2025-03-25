import visdom
import numpy as np

# Connect to Visdom
vis = visdom.Visdom()

# Check if Visdom server is running
if not vis.check_connection():
    print("Visdom server is not running. Please start it.")
else:
    # Example: Visualize a simple line plot
    X = np.linspace(0, 2 * np.pi, 100)
    Y = np.sin(X)

    # Create a line plot
    vis.line(Y, X, win='sin_wave', opts=dict(title='Sine Wave', xlabel='X', ylabel='Y'))
    
    # Example: Visualize an image (random image for demonstration)
    img = np.random.rand(3, 256, 256)  # 3x256x256 image
    vis.image(img, opts=dict(title='Random Image'))
