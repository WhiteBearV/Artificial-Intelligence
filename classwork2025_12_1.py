import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf

mist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mist.load_data()

#printing the shape
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

#Displaying the first 9 image of dataset
fig = plt.figure(figsize=(10,10))

nrow=3
ncol=3
for i in range(9):
    fig.add_subplot(nrow, ncol, i+1)
    plt.imshow(train_images[i])
    plt.title(f"Digit: {train_labels[i]}")
    plt.axis(False)
plt.show()

#Covertig image pixel values to 0-1 
train_images = train_images / 255.0
test_images = test_images / 255.0

print("First Label before coversion:", train_labels[0])
print(train_labels[0])

#Converting labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

print("First Label after coversion:")
print(train_labels[0])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),                      # 28x28 -> 784
    tf.keras.layers.Dense(10, activation='softmax') # 10 classes
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    x = train_images,
    y = train_labels,
    epochs = 10
)

#Showing plot for loss
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.legend('loss')
plt.show()

#Showing plot for accuracy
plt.plot(history.history['accuracy'], color='orange')
plt.xlabel('epochs')
plt.legend('accuracy')
plt.show()

#Call evaluate to  find accuracy on test images
test_loss, test_accuracy = model.evaluate(
    x = test_images, 
    y = test_labels
    )

print("Test loss: %.4f"%test_loss)
print("Test accuracy: %.4f"%test_accuracy)

# === Part 7 === #

predicted_probabilities = model.predict(test_images)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()

index=11

# Showing image
plt.imshow(test_images[index])

# Printing Probabilities
print("Probabilities predicted for image at index", index)
print(predicted_probabilities [index])

print()

# Printing Predicted Class
print("Probabilities class for image at index", index)
print(predicted_classes [index])
