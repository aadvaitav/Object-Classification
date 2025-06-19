import tensorflow as tf
from tensorflow.keras import layers, datasets, utils
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load and preprocess dataset
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode labels
train_labels = utils.to_categorical(train_labels, 10)
test_labels = utils.to_categorical(test_labels, 10)

# Check if model already exists
if os.path.exists('cifar10_model.h5'):
    print("Loading existing model...")
    model = tf.keras.models.load_model('cifar10_model.h5')
else:
    print("Training new model...")
    
    # Simple CNN model
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2
    )

    # Save model
    model.save('cifar10_model.h5')
    print("Model saved!")

# Test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")

# CIFAR-10 class names
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Predict on your own image
print("\n" + "="*50)
print("PREDICT YOUR OWN IMAGE")
print("="*50)

# Change this path to your image
image_path = "path to your image"

# Check if image exists
if os.path.exists(image_path):
    # Load and predict
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print(f"Prediction: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.1f}%")
else:
    print(f"Image not found: {image_path}")
    print("Please put your image in the same folder and update the path")

# Test on sample CIFAR-10 images
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

# Show predictions for first 5 test images
sample_predictions = model.predict(test_images[:5], verbose=0)
actual_labels = np.argmax(test_labels[:5], axis=1)

for i in range(5):
    predicted = np.argmax(sample_predictions[i])
    actual = actual_labels[i]
    confidence = np.max(sample_predictions[i]) * 100
    
    print(f"Image {i+1}: Predicted={class_names[predicted]}, Actual={class_names[actual]}, Confidence={confidence:.1f}%")
