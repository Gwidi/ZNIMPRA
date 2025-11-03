from argparse import ArgumentParser
import tensorflow as tf

import ai


def main(args):
    # todo: load mnist dataset
    train_ds, val_ds = ai.datasets.mnist(512)

    # todo: create and optimize model (add regularization like dropout and batch normalization)
    model = ai.models.image.ImageClassifier(num_classes=10)

    # todo: create optimizer (optional: try with learning rate decay)
    optimizer = tf.optimizers.Adam(0.0001)

    # todo: define query function
    def query(batch, training):
        images, labels = batch
        images = tf.expand_dims(images, -1)  # Add channel dimension: (batch_size, 28, 28) -> (batch_size, 28, 28, 1)
        images = tf.cast(images, tf.float32) / 255.0  # Convert to float32 and normalize to [0, 1]
        y_pred = model(images, training=training)
        loss = ai.losses.classification.classification_loss(labels, y_pred)

        return loss

    # todo: define train function
    def train(batch):
        with tf.GradientTape() as tape: # Gradient tape records operations to compute gradients later
            loss = query(batch, True)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    # todo: run training and evaluation for number or epochs (from argument parser)
    #  and print results (accumulated) from each epoch (train and val separately)
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')

        train_loss = tf.metrics.Mean('train_loss')
        for batch in train_ds:
            loss = train(batch)
            train_loss.update_state(loss)

        val_loss = tf.metrics.Mean('val_loss')
        for batch in val_ds:
            loss = query(batch, False)
            val_loss.update_state(loss)

        print(f'Train Loss: {train_loss.result().numpy():.4f}, Val Loss: {val_loss.result().numpy():.4f}')

        


if __name__ == '__main__':
    parser = ArgumentParser()
    # todo: pass arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--allow-memory-growth', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        ai.utils.allow_memory_growth()

    main(args)
