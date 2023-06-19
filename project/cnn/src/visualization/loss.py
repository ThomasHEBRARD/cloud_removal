import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("models/epochs=500,lr=0.0001,batch_size=16,nb_batches_per_epoch=1000,model_path=,loss=mse/loss_log.txt", sep=",", header=None)
df2 = pd.read_csv("models/continue_epochs=500,lr=0.0001,batch_size=16,nb_batches_per_epoch=1000,model_path=,loss=mse/loss_log.txt", sep=",", header=None)
df.columns = ["id", 'loss', 'validation_loss', 'accuracy', 'validation_accuracy']
df2.columns = ["id", 'loss', 'validation_loss', 'accuracy', 'validation_accuracy']

df = pd.concat([df, df2])
df = df.reset_index(drop=True)
# Assuming 'df' is your DataFrame and it has columns 'loss', 'accuracy', 'validation_loss', 'validation_accuracy'

plt.figure(figsize=(12, 6))

# Plot training loss and validation loss
plt.subplot(1, 1, 1)
# plt.plot(df['loss'], label='Training Loss')
# plt.plot(df['validation_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig("valcnn.png")

# Plot training accuracy and validation accuracy
plt.subplot(1, 1, 1)
plt.plot(df['accuracy'], label='Training Accuracy')
plt.plot(df['validation_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("tracnn.png")

plt.tight_layout()
# plt.show()
