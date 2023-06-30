import pandas as pd
from matplotlib import pyplot as plt

run_name = "epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy"

path = f"models/{run_name}/"
d_loss_path = f"/Users/thomashebrard/thesis/code/project/pix2pix/models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy/model_epoch_175/model_epoch_175.h5,v=cloudy/d_loss.txt"
g_loss_path = f"/Users/thomashebrard/thesis/code/project/pix2pix/models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy/model_epoch_175/model_epoch_175.h5,v=cloudy/g_loss.txt"
val_loss_path = f"/Users/thomashebrard/thesis/code/project/pix2pix/models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy/model_epoch_175/model_epoch_175.h5,v=cloudy/val_loss2.txt"

g_loss = pd.read_csv(g_loss_path, sep=",", header=None)
g_loss.columns = ['idx', 'val']
g_loss = g_loss.groupby('idx').mean().reset_index()
g_loss['values_smoothed'] = g_loss['val'].rolling(window=20).mean()
g_loss['val']= g_loss['values_smoothed']


d_loss = pd.read_csv(d_loss_path, sep=",", header=None)
d_loss.columns = ['idx', 'val']
d_loss = d_loss.groupby('idx').mean().reset_index()
d_loss['values_smoothed'] = d_loss['val'].rolling(window=20).mean()
d_loss['val']= d_loss['values_smoothed']


val_loss = pd.read_csv(val_loss_path, sep=",", header=None)
val_loss.columns = ['idx', 'val1', "val2", "val3"]
val_loss = val_loss.groupby('idx').mean().reset_index()
val_loss['values_smoothed'] = val_loss['val1'].rolling(window=20).mean()

fig, ax1 = plt.subplots()

# Plot the data from g_loss on ax1
ax1.plot(g_loss['idx'], g_loss['val'], color='blue')
ax1.set_ylabel('g_loss', color='blue')
ax1.tick_params('y', colors='blue')

# Create ax2, which shares the x-axis with ax1
# ax2 = ax1.twinx()

# # Plot the data from d_loss on ax2
# ax2.plot(d_loss['idx'], d_loss['val'], color='red')
# ax2.set_ylabel('d_loss', color='red')
# ax2.tick_params('y', colors='red')

# Create ax3, which shares the x-axis with ax1
ax3 = ax1.twinx()

# Offset the ax3 spine (the line noting the data range for the ax3 plot)
ax3.spines['right'].set_position(('outward', 60))  

# Plot the data from val_loss on ax3
ax3.plot(val_loss['idx'], val_loss['val1'], color='green')  # Replace 'val1' with the column name you want to plot
ax3.set_ylabel('val_loss', color='green')
ax3.tick_params('y', colors='green')

# Set the x-axis label
ax1.set_xlabel('epoch')

plt.title('Generator loss, Discriminator loss and Validation loss of GAN')
plt.savefig("loss.png")
fig.tight_layout()

# plt.show()

plt.plot(val_loss['idx'], val_loss['val1'], label='Validation loss')
plt.plot(g_loss['idx'], g_loss['val'], label='Generator loss')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plotting Two Arrays')

# Adding legend
plt.legend()

# Displaying the plot
plt.show()


########################################################################################################################

# import pandas as pd
# from matplotlib import pyplot as plt

# run_name = "epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy"

# path = f"models/{run_name}/"
# val_loss_path = f"/Users/thomashebrard/thesis/code/project/pix2pix/models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy/model_epoch_175/model_epoch_175.h5,v=cloudy/val_loss2.txt"

# val_loss = pd.read_csv(val_loss_path, sep=",", header=None)
# val_loss.columns = ['idx', 'val1', "val2", "val3"]

# fig, ax1 = plt.subplots()

# # Plot the data for val1
# ax1.plot(val_loss['idx'], val_loss['val1'], color='blue')
# ax1.set_ylabel('val1', color='blue')
# ax1.tick_params('y', colors='blue')

# # Create ax2, which shares the x-axis with ax1
# ax2 = ax1.twinx()
# # Plot the data for val2
# ax2.plot(val_loss['idx'], val_loss['val2'], color='red')
# ax2.set_ylabel('val2', color='red')
# ax2.tick_params('y', colors='red')

# # Create ax3, which shares the x-axis with ax1
# ax3 = ax1.twinx()
# # Offset the right spine of ax3. The ticks and label have already been placed on the right by twinx()
# ax3.spines['right'].set_position(('outward', 60))
# # Plot the data for val3
# ax3.plot(val_loss['idx'], val_loss['val3'], color='green')
# ax3.set_ylabel('val3', color='green')
# ax3.tick_params('y', colors='green')

# # Set the x-axis label
# ax1.set_xlabel('epoch')

# plt.title('Value 1, Value 2, and Value 3')
# plt.savefig("val_loss.png")
# fig.tight_layout()

# plt.show()
# ################################################################################################################################################

# import pandas as pd
# from matplotlib import pyplot as plt

# run_name = "epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy"

# path = f"models/{run_name}/"
# val_loss_path = f"/Users/thomashebrard/thesis/code/project/pix2pix/models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy/model_epoch_175/model_epoch_175.h5,v=cloudy/val_loss2.txt"

# val_loss = pd.read_csv(val_loss_path, sep=",", header=None)
# val_loss.columns = ['idx', 'val1', "val2", "val3"]
# val_loss = val_loss.groupby('idx').mean().reset_index()
# fig, axs = plt.subplots(3)

# # Plot the data for val1
# axs[0].plot(val_loss['idx'], val_loss['val1'], color='blue')
# axs[0].set_ylabel('val1')

# # Plot the data for val2
# axs[1].plot(val_loss['idx'], val_loss['val2'], color='red')
# axs[1].set_ylabel('val2')

# # Plot the data for val3
# axs[2].plot(val_loss['idx'], val_loss['val3'], color='green')
# axs[2].set_ylabel('val3')

# # Set the x-axis label
# axs[2].set_xlabel('epoch')

# plt.tight_layout()
# plt.show()
