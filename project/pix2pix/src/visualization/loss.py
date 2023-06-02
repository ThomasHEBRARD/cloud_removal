import pandas as pd
from matplotlib import pyplot as plt

path =  "models/run_2023-06-01T00:22:32/"
d_loss_path = f"{path}d_loss.txt"
g_loss_path = f"{path}g_loss.txt"


g_loss = pd.read_csv(g_loss_path, sep=",", header=None)
g_loss.columns = ['idx', 'val']

d_loss = pd.read_csv(d_loss_path, sep=",", header=None)
d_loss.columns = ['idx', 'val']

fig, ax1 = plt.subplots()

# Plot the data from df1 on ax1
ax1.plot(g_loss['idx'], g_loss['val'], color='blue')
ax1.set_ylabel('g_loss', color='blue')
ax1.tick_params('y', colors='blue')

# Create ax2, which shares the x-axis with ax1
ax2 = ax1.twinx()

# Plot the data from df2 on ax2
ax2.plot(d_loss['idx'], d_loss['val'], color='red')
ax2.set_ylabel('d_loss', color='red')
ax2.tick_params('y', colors='red')

# Set the x-axis label
ax1.set_xlabel('ID')

plt.title('ID vs Value for both DataFrames')
fig.tight_layout()

plt.show()