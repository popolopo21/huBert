import keras
import tensorflow as tf


class MLModel(tf.keras.Model):
    #Load the loss function
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)
    #loss calculated

    loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, inputs):
        if len(inputs) == 3:

            # features = masked_encoded_texts, labels = encoded_text, sample_weight = sample_weight
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        #Optimizing the model
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = self.loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # We have to define metrics property, what the model can call after each epoch to reset the states

        return [self.loss_tracker]
