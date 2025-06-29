# test_resume.py
import numpy as np
import tensorflow as tf

# Load your existing model
model_path = "model_20250408_005934.keras"
model = tf.keras.models.load_model(model_path)

# Extract weights
weights = []
for layer in model.layers:
    layer_weights = layer.get_weights()
    for w in layer_weights:
        weights.extend(w.flatten())

print(f"Extracted {len(weights)} weight values")
print(f"Weight range: {min(weights):.4f} to {max(weights):.4f}")

# These weights could be used as individual[0] in your GA population