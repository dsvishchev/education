import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных MNIST
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
# Нормализуем ввод в диапазоне [-1, 1]
train_images = (train_images.astype(np.float32) - 127.5) / 127.5
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

# Гиперпараметры
latent_dim = 100
generator_input_shape = (latent_dim,)
discriminator_input_shape = (28, 28, 1)


# Вспомогательная функция для отображения сгенерированных изображений
def plot_generated_images(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=(examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(10):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')


# Генератор
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 128, input_shape=generator_input_shape))  # принимает шум, возвращает вектор
    model.add(layers.Reshape((7, 7, 128)))  # преобразоване в тензор
    model.add(layers.BatchNormalization())  # нормализация пакета
    model.add(layers.UpSampling2D())  # увеличение размерности в 2 раза
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu'))  # сверточный слой
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
    return model


# Дискриминатор
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=discriminator_input_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    # Слой исключения с вероятностью исключения 30 %, который
    # регуляризует данные, исключая случайно выбранные нейроны
    # во время обучения
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten()) # преобразует в вектор
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# Создание и компиляция моделей
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False

gan_input = tf.keras.Input(shape=generator_input_shape)
x = generator(gan_input)
gan_output = discriminator(x)

gan = models.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')


# Обучение GAN
def train_GAN(epochs, batch_size):
    batch_count = train_images.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):
            # Создаем случайный шум для генератора
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            # Генерируем изображения с помощью генератора
            generated_images = generator.predict(noise)

            # Выбираем случайный набор настоящих изображений для дискриминатора
            image_batch = train_images[np.random.randint(0, train_images.shape[0], batch_size)]

            # Создаем входные данные для дискриминатора
            X = np.concatenate([image_batch, generated_images])
            # Создаем метки для дискриминатора
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 1

            # Обучаем дискриминатор на входных данных и метках
            discriminator_loss = discriminator.train_on_batch(X, y_dis)

            # Создаем новый случайный шум для генератора
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            # Создаем метки для генератора
            y_gen = np.ones(batch_size)

            # Обучаем генератор на шуме и метках
            generator_loss = gan.train_on_batch(noise, y_gen)

        print(f"Epoch {epoch + 1}, D Loss: {discriminator_loss[0]}, G Loss: {generator_loss}")

        if (epoch + 1) % 10 == 0:
            plot_generated_images(generator, epoch + 1, latent_dim)


# Обучение GAN
train_GAN(epochs=10, batch_size=128)

# Сохранение генератора
generator.save('generator_model.h5')

# Сохранение дискриминатора
discriminator.save('discriminator_model.h5')

gan.save('gan_model.h5')

# Загрузка моделей
loaded_generator = tf.keras.models.load_model('generator_model.h5')
loaded_discriminator = tf.keras.models.load_model('discriminator_model.h5')
loaded_gan = tf.keras.models.load_model('gan_model.h5')
latent_dim = 100

plot_generated_images(loaded_generator, '', latent_dim)