import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Load and preprocess data
input_data = np.loadtxt('input_data1.txt').reshape(-1, 256)  # (10000, 256)
target_data = np.loadtxt('target_data1.txt').reshape(-1, 256)  # (10000, 256)

# Split data into train, validation, and test sets
x_train, x_val, x_test = input_data[:6000], input_data[6000:8000], input_data[8000:]
y_train, y_val, y_test = target_data[:6000], target_data[6000:8000], target_data[8000:]

# Define model
model = Sequential([
    Dense(1024, activation='relu', input_shape=(256,)),
    BatchNormalization(),
    Dropout(0.2),
    LayerNormalization(),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    LayerNormalization(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    LayerNormalization(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(256, activation='linear')  # Output layer for regression
])

# Compile model
model.compile(optimizer=RMSprop(learning_rate=0.00001), loss='mse', metrics=['mae'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-8)

# Train model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=5000,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate on test set
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f'Test Loss (MSE): {test_loss}')
print(f'Test MAE: {test_mae}')

# Plot training & validation loss values
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error (MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()
