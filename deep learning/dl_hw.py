import tensorflow as tf
import numpy as np
import argparse
import logging

# parse arguments -e epochs -v verbose -p pretrained -r rate
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=10000, help="number of epochs")
parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
parser.add_argument("-p", "--pretrained", action="store_true", help="pretrained model")
parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")

args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
log.info("Logger started")
log.info("Arguments: " + str(args))

# Define the input and output data
input_data = np.array([[0, 0, 0],
                       [1, 0.67, 0.33], # 0.25
                       [.5, 1.0, .5],
                       [.33, .67, 1],
                       # [0, 0, 0]
                       ]).astype(np.float32)
output_data = np.array([
    [0],
    [.25],
    [.5],
    [.75],
    #[1]
    ]).astype(np.float32)

if args.pretrained:
    log.debug("Loading pretrained model")
    model = tf.keras.models.load_model('my_model.h5')
else:
    # Define the model architecture
    log.debug("Creating new model")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=9, input_shape=(3,), activation='relu'),
        tf.keras.layers.Dense(units=9, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='tanh')
    ])

    log.debug("Compiling model")
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(input_data, output_data, epochs=args.epochs, verbose=args.verbose, )
    
    log.debug("Saving model")
    model.save('my_model.h5')

# Define some new input data for prediction
new_input_data = np.array([
    [1, 0.67, 0.33],  # exactly 0.25
    [1.1, 0.57, 0.33],
    [0.3, 0.9, 0.7],
    [0.4, 0.75, 0.9],
    [1, 0.67, 0.33], # 0.25
    [.5, 1.0, .5],
    [.33, .67, 1],
    [0, 0, 0]
    ]).astype(np.float32)

# Predict with the trained model
predictions = model.predict(new_input_data)
# should be approximately [[1], [2], [3]]

# print out weights
# print(model.get_weights())

# Print the predictions
print(predictions)
