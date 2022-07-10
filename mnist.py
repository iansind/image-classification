# Training of a ML model on the MNIST handwritten digit database using TensorFlow. 
# Part of a course in computer vision at CU Boulder. 
# Greater than 98% testing accuracy obtained. 

import tensorflow as tf

mnist = tf.keras.datasets.mnist
tf.keras.backend.set_floatx('float64')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('\nTraining shape: ', x_train.shape)
print('\nTesting shape: ', x_test.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(392, activation='relu'), # Non-linear activation function added to dense layer to improve accuracy.
    tf.keras.layers.Dense(10),
])

# Gets an array of logit scores for each class.
predictions = model(x_train[:1]).numpy()
print('/nNon-normalized predictions: ', predictions)

# Softmax to convert logits to workable probabilities for each class.
normalized_predictions = tf.nn.softmax(predictions).numpy()
print('/nNormalized predictions: ', normalized_predictions)

# Determination of a scalar loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(y_train[:1], predictions).numpy()
print('\nUntrained loss: ', loss)

# Training a model with 10 epochs.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('\nTest accuracy: ', test_acc)
