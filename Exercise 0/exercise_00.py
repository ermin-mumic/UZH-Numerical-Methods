import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg') # Backend for interactive plots

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figwidth(12)
fig.set_figheight(5)

# Vectors as 1D NumPy arrays
vf = np.array([1., 0.]) # Forward vector [x=1, y=0]
vb = np.array([1., 0.]) # Backward vector [x=1, y=0]

# 5° rotation angle in radians
angle = 5. * np.pi / 180.

# 2D rotation matrix
A = np.array([[np.cos(angle), np.sin(angle)],
              [-np.sin(angle), np.cos(angle)]])

# Create arrow plots (quiver plots)
qf = ax1.quiver(0, 0, 1, 0)  # Arrow from (0,0) to (1,0)
qb = ax2.quiver(0, 0, 1, 0)


def init():
    # Set up both plots
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True)
    ax1.set_title("Chart 1")
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True)
    ax2.set_title("Chart 2")
    return None,


init()
plt.show()
