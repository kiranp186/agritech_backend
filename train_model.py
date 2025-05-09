import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    validation_split=0.2
)

# Load data
train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)

# OPTIONAL: Save original model
model.save('model.h5')

# ✅ Convert to quantized TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
tflite_quant_model = converter.convert()

# ✅ Save the quantized TFLite model
with open("model_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Quantized model saved as model_quant.tflite")
