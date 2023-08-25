import tensorflow as tf
import numpy as np

# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 127.5 - 1.0  # Normalize to range [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

# Generator model
def build_generator(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(np.prod(output_shape), activation='tanh'),
        tf.keras.layers.Reshape(output_shape)
    ])
    return model

# Discriminator model
def build_discriminator(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Define input shapes
input_shape = (100,)  # Random noise vector input
output_shape = (28, 28, 1)  # MNIST image shape

# Build and compile the models
generator = build_generator(input_shape, output_shape)
discriminator = build_discriminator(output_shape)
gan = build_gan(generator, discriminator)

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Training parameters
num_epochs = 10000
batch_size = 64
steps_per_epoch = x_train.shape[0] // batch_size
discriminator_iterations = 1

# Training loop
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        # Train discriminator
        discriminator.trainable = True
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        noise = np.random.normal(0, 1, (batch_size, *input_shape))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, *input_shape))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
    # Print and record losses, update plots, etc.
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Generate and save example images
num_examples = 10
noise = np.random.normal(0, 1, (num_examples, *input_shape))
generated_images = generator.predict(noise)
generated_images = 0.5 * generated_images + 0.5  # Rescale images to [0, 1]

# Save generated images
for i in range(num_examples):
    img = generated_images[i]
    img = np.squeeze(img)
    img = (img * 255).astype(np.uint8)
    tf.keras.preprocessing.image.save_img(f"generated_image_{i}.png", img)
