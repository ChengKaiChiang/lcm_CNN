import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# data preprocess - do normalization
def data_preprocess(data):
    # Normalization
    data['NTSC'] = data['NTSC'] / 99
    data['brightness'] = data['brightness'] / 700
    data['gamma'] = (data['gamma'] - 1.8) / (2.6 - 1.8)
    data['temperature'] = (data['temperature'] - 1000) / (10000 - 1000)
    return data

# neural network model - simplify model size, but if you want more layer, you can restore to your origin model
# In addition, the output is five elements. Because you should classify each element by probability.
def build():
    model = tf.keras.Sequential(name='LCM')
    model.add(tf.keras.layers.InputLayer(input_shape = (4,)))
    model.add(tf.keras.layers.Reshape((4, 1)))
    model.add(tf.keras.layers.Conv1D(filters = 16, strides = 1, padding = 'same', kernel_size = 2, activation = 'sigmoid'))
    model.add(tf.keras.layers.Conv1D(filters = 32, strides = 1, padding = 'same', kernel_size = 2, activation = 'sigmoid'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size = 2, strides = 1, padding = 'same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(5))
    return model

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()

# Load dataset
# Why we minus 1 in y ? Because the output range of model is [0, 4] 
dataset = pd.read_csv('train.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1] - 1

x = data_preprocess(x)
x = x.values
y = y.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

print("\n\n\n============================================== Train ==============================================")
model = eval("build()")
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size = 256, epochs = 100, verbose = 1, validation_data = (x_test, y_test))

show_train_history(history, 'accuracy', 'val_accuracy')
print("============================================ Evaluate =============================================")
model.evaluate(x_test, y_test)


print("\n\n\n============================================= Predict =============================================")
# Predict Example
print("This is a test input: ", end = '')
test_input = x_test[11]
print(test_input)
print("")

print("This is a test output: <Level ", end = '')
test_output = y_test[11] + 1
print(test_output, end = '')
print(">\n\n")

predict_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = predict_model.predict(np.reshape(test_input, (1, 4)))
print("The probability of each elements: ", end = '')
print(np.around(predictions[0] * 100, 2))

print("Level is ", end = '')
print(np.argmax(predictions) + 1)