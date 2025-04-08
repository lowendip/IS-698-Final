import tensorflow as tf
import numpy as np
import
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

NUM_CLIENTS = 10
client_data = []
data_per_client = len(x_train) // NUM_CLIENTS
for i in range(NUM_CLIENTS):
    start = i * data_per_client
    end = (i + 1) * data_per_client
    client_data.append((x_train[start:end], y_train[start:end]))

NUM_ROUNDS = 10
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    print(f'Round {round_num+1}, metrics={metrics}')

test_loss, test_acc = global_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')