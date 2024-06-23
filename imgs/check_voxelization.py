import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Axes3D import is used to register the 3D projection

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # This is a more common way to create a 3D axis

# Load your voxel data
x = np.load("voxel_grid_flat_fin_h.npy")
print(x)
print(f"x.shape: {x.shape}")

# Check data
if np.count_nonzero(x) == 0:
    print("The voxel data is empty or mostly zero.")

# Customize colors and transparency
ax.voxels(x, facecolors='blue', edgecolor='k')  # 'k' stands for black edge color

# Adjust the viewing angle
ax.view_init(elev=20, azim=30)  # 'elev' and 'azim' are elevation and azimuthal angles

# Set a background color that contrasts with your voxel colors
ax.set_facecolor('white')  # This sets the background color


# Adding axis labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')


plt.show()

