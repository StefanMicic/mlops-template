import tensorflow as tf
from keras.layers import Dense
from tensorflow import keras


class Model(keras.Model):
    """A model for test classification. It should tell whether the test is positive or not."""

    def __init__(self):
        """Creates instance of test classification model."""
        super(Model, self).__init__()
        self.input_layer = Dense(12, input_dim=8, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass in neural network.

        Args:
            inputs: Input tensor.

        Returns:
            Prediction given by model.
        """
        return self.output_layer(self.input_layer(inputs))
