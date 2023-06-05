import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("t.txt", sep=",", header=None)
df.columns = ['loss', 'accuracy', 'validation_loss', 'validation_accuracy']
# Assuming 'df' is your DataFrame and it has columns 'loss', 'accuracy', 'validation_loss', 'validation_accuracy'

plt.figure(figsize=(12, 6))

# Plot training loss and validation loss
plt.subplot(1, 2, 1)
plt.plot(df['loss'], label='Training Loss')
plt.plot(df['validation_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(df['accuracy'], label='Training Accuracy')
plt.plot(df['validation_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
