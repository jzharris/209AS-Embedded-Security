from keras.models import Sequential, Model, load_model
from keras.layers import UpSampling2D, Conv2D, Activation, BatchNormalization, Reshape, Dense, Input, LeakyReLU, Dropout, Flatten, ZeroPadding2D
from keras.optimizers import Adam
import keras

import glob
from PIL import Image
import numpy as np
import os
import argparse
from ast import literal_eval
import cv2

import imageio


class BlackBox:
    def __init__(self, discriminator_path, generator_path, output_directory, img_size, img_channels, builtin_dataset):
        self.use_cifar = builtin_dataset == "cifar10"
        self.use_mnist = builtin_dataset == "mnist"
        if self.use_cifar or self.use_mnist:
            self.img_size = (32, 32)
            self.channels = 3 if self.use_cifar else 1
        else:
            self.channels = img_channels
            self.img_size = img_size

        self.upsample_layers = 5
        self.starting_filters = 64
        self.kernel_size = 3
        self.discriminator_path = discriminator_path
        self.generator_path = generator_path
        if builtin_dataset:
            orig_dir = output_directory.split("/")
            output_directory = ""
            for folder in orig_dir[:-1]:
                output_directory += folder + "/"
            output_directory += builtin_dataset
        self.output_directory = output_directory

        self.blackbox_discriminator = None
        self.blackbox_estimator = None

    def build_blackbox_model(self):
        """
        Builds the entire blackbox model (beginning and end and known to us)
        :return:
        """

        black_model = Sequential()
        black_model = self.build_discriminator_beginning(black_model)
        black_model = self.build_discriminator_blackbox(black_model)
        black_model = self.build_discriminator_end(black_model)

        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        black_model.summary()
        img = Input(shape=img_shape)
        validity = black_model(img)

        return Model(img, validity)

    def build_estimator_model(self):
        """
        Builds the entire estimator model (using middle portion to estimate middle portion of blackbox model)
        :return:
        """

        estim_model = Sequential()
        estim_model = self.build_discriminator_beginning(estim_model)
        estim_model = self.build_estimator_of_blackbox(estim_model)
        estim_model = self.build_discriminator_end(estim_model)

        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        estim_model.summary()
        img = Input(shape=img_shape)
        validity = estim_model(img)

        return Model(img, validity)

    def build_discriminator_beginning(self, model):
        """
        The beginning portion of the discriminator
        :return:
        """
        assert model is not None
        img_shape = (self.img_size[0], self.img_size[1], self.channels)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=self.kernel_size, strides=2, input_shape=img_shape,
                              padding="same"))  # 192x256 -> 96x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        return model

    def build_discriminator_blackbox(self, model):
        """
        We are unaware of this middle portion, and will use another NN to approximate it.
        :return:
        """
        assert model is not None

        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=2, padding="same"))  # 96x128 -> 48x64
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))  # 48x64 -> 24x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=1, padding="same"))  # 24x32 -> 12x16
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, kernel_size=self.kernel_size, strides=1, padding="same"))  # 12x16 -> 6x8
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        return model

    def build_estimator_of_blackbox(self, model):
        """
        NN layers used to approximate blackbox portion.
        :return:
        """
        assert model is not None

        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=2, padding="same"))  # 96x128 -> 48x64
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))  # 48x64 -> 24x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=1, padding="same"))  # 24x32 -> 12x16
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, kernel_size=self.kernel_size, strides=1, padding="same"))  # 12x16 -> 6x8
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        return model

    @staticmethod
    def build_discriminator_end(model):
        assert model is not None

        model.add(Flatten())
        model.add(Dense(10, activation='sigmoid'))

        return model

    def build_gan(self):
        """
        Build the two-model system
        :return:
        """
        optimizer = Adam(0.0002, 0.5)

        # Build the blackbox discriminator:
        # The discriminator_path MUST have pretrained weights in this experiment.
        assert os.path.exists(self.discriminator_path)
        self.blackbox_discriminator = load_model(self.discriminator_path)

        # Build the blackbox estimator:
        self.blackbox_estimator = self.build_estimator_model()
        self.blackbox_estimator.compile(loss='binary_crossentropy',
                                        optimizer=optimizer)

        # These next few lines setup the training for the GAN model
        z = Input(shape=(100,))
        img = self.generator(z)

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


class DCGAN:
    def __init__(self, discriminator_path, generator_path, output_directory, img_size, img_channels, builtin_dataset):
        self.use_cifar = builtin_dataset == "cifar10"
        self.use_mnist = builtin_dataset == "mnist"
        if self.use_cifar or self.use_mnist:
            self.img_size = (32, 32)
            self.channels = 3 if self.use_cifar else 1
        else:
            self.channels = img_channels
            self.img_size = img_size

        self.upsample_layers = 5
        self.starting_filters = 64
        self.kernel_size = 3
        self.discriminator_path = discriminator_path
        self.generator_path = generator_path
        if builtin_dataset:
            orig_dir = output_directory.split("/")
            output_directory = ""
            for folder in orig_dir[:-1]:
                output_directory += folder + "/"
            output_directory += builtin_dataset
        self.output_directory = output_directory

    def build_generator(self):
        noise_shape = (100,)

        # This block of code can be a little daunting, but essentially it automatically calculates the required starting
        # array size that will be correctly upscaled to our desired image size.
        #
        # We have 5 Upsample2D layers which each double the images width and height, so we can determine the starting
        # x size by taking (x / 2^upsample_count) So for our target image size, 256x192, we do the following:
        # x = (192 / 2^5), y = (256 / 2^5) [x and y are reversed within the model]
        # We also need a 3rd dimension which is chosen relatively arbitrarily, in this case it's 64.
        model = Sequential()
        model.add(
            Dense(self.starting_filters * (self.img_size[0] // (2 ** self.upsample_layers))  *  (self.img_size[1] // (2 ** self.upsample_layers)),
                  activation="relu", input_shape=noise_shape))
        model.add(Reshape(((self.img_size[0] // (2 ** self.upsample_layers)),
                           (self.img_size[1] // (2 ** self.upsample_layers)),
                           self.starting_filters)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 6x8 -> 12x16
        model.add(Conv2D(1024, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 12x16 -> 24x32
        model.add(Conv2D(512, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 24x32 -> 48x64
        model.add(Conv2D(256, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 48x64 -> 96x128
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 96x128 -> 192x256
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(32, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_size[0], self.img_size[1], self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=self.kernel_size, strides=2, input_shape=img_shape, padding="same"))  # 192x256 -> 96x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=2, padding="same"))  # 96x128 -> 48x64
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))  # 48x64 -> 24x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=1, padding="same"))  # 24x32 -> 12x16
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, kernel_size=self.kernel_size, strides=1, padding="same"))  # 12x16 -> 6x8
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_gan(self):
        optimizer = Adam(0.0002, 0.5)

        # See if the specified model paths exist, if they don't then we start training new models

        if os.path.exists(self.discriminator_path) and os.path.exists(self.generator_path):
            self.discriminator = load_model(self.discriminator_path)
            self.generator = load_model(self.generator_path)
            print("Loaded models...")
        else:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=optimizer,
                                       metrics=['accuracy'])

            self.generator = self.build_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # These next few lines setup the training for the GAN model
        z = Input(shape=(100,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load_imgs(self, image_path):
        if self.use_cifar:
            (x_train, train_labels), (_, _) = keras.datasets.cifar10.load_data()
        elif self.use_mnist:
            x_train = []
            (imgs, train_labels), (_, _) = keras.datasets.mnist.load_data()
            print(imgs.shape)
            for img in imgs:
                # scale image to be the same WxH as we need:
                img = cv2.resize(img, dsize=(self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_CUBIC)
                # also add channels:
                if len(img.shape) <= 2:
                    img = img[:, :, np.newaxis]
                x_train.append(img)
        else:
            x_train = []
            for i in glob.glob(image_path):
                img = Image.open(i)
                img = np.asarray(img)
                if img.shape != self.img_size:
                    # scale image to be the same WxH as we need:
                    img = cv2.resize(img, dsize=(self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_CUBIC)
                    # also add channels:
                    if len(img.shape) <= 2:
                        img = img[:, :, np.newaxis]
                x_train.append(img)
        return np.asarray(x_train)

    def train(self, epochs, image_path, batch_size=32, save_interval=50):
        self.build_gan()
        x_train = self.load_imgs(image_path)
        print("Training Data Shape: ", x_train.shape)

        # Rescale images from -1 to 1
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        half_batch = batch_size // 2

        for epoch in range(epochs):


            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))



            # Train Discriminator
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Print progress
            print(f"{epoch} [D loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

            # If at save interval => save generated image samples, save model files
            if epoch % (save_interval) == 0:

                self.save_imgs(epoch)

                save_path = self.output_directory + "/models"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.discriminator.save(save_path + "/discrim.h5")
                self.generator.save(save_path + "/generat.h5")

    def gene_imgs(self, count):
        # Generate images from the currently loaded model
        noise = np.random.normal(0, 1, (count, 100))
        return self.generator.predict(noise)

    def save_imgs(self, epoch):
        r, c = 5, 5

        # Generates r*c images from the model, saves them individually and as a gallery

        imgs = self.gene_imgs(r*c)
        imgs = 0.5 * imgs + 0.5

        for i, img_array in enumerate(imgs):
            path = f"{self.output_directory}/generated_{self.img_size[0]}x{self.img_size[1]}"
            if not os.path.exists(path):
                os.makedirs(path)
            imageio.imwrite(path + f"/{epoch}_{i}.png", img_array)

        nindex, height, width, intensity = imgs.shape
        nrows = nindex // c
        assert nindex == nrows * c
        # want result.shape = (height*nrows, width*ncols, intensity)
        gallery = (imgs.reshape(nrows, c, height, width, intensity)
                  .swapaxes(1, 2)
                  .reshape(height * nrows, width * c, intensity))

        path = f"{self.output_directory}/gallery_generated_{self.img_size[0]}x{self.img_size[1]}"
        if not os.path.exists(path):
            os.makedirs(path)
        imageio.imwrite(path + f"/{epoch}.png", gallery)

    def generate_imgs(self, count, threshold, modifier):
        self.build_gan()

        # Generates (count) images from the model ensuring the discriminator scores them between the threshold values
        # and saves them

        imgs = []
        for i in range(count):
            score = [0]
            while not(threshold[0] < score[0] < threshold[1]):
                img = self.gene_imgs(1)
                score = self.discriminator.predict(img)
            print("Image found: ", score[0])
            imgs.append(img)

        imgs = np.asarray(imgs).squeeze()
        imgs = 0.5 * imgs + 0.5

        print(imgs.shape)
        for i, img_array in enumerate(imgs):
            path = f"{self.output_directory}/generated_{threshold[0]}_{threshold[1]}"
            if not os.path.exists(path):
                os.makedirs(path)
            imageio.imwrite(path + f"/{modifier}_{i}.png", img_array)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_generator', help='Path to existing generator weights file', default="./data/models/generat.h5")
    parser.add_argument('--load_discriminator', help='Path to existing discriminator weights file', default="./data/models/discrim.h5")
    parser.add_argument('--data', help='Path to directory of images of correct dimensions, using *.[filetype] (e.g. *.png) to reference images', default="./data/att-database-of-faces/*/*.pgm")
    parser.add_argument('--sample', help='If given, will generate that many samples from existing model instead of training', default=-1)
    parser.add_argument('--sample_thresholds', help='The values between which a generated image must score from the discriminator', default="(0.0, 0.1)")
    parser.add_argument('--batch_size', help='Number of images to train on at once', default=48)    #24: reg, 48:mnist and cifar
    parser.add_argument('--builtin_dataset', default="cifar10")
    parser.add_argument('--image_size', help='Size of images as tuple (height,width). Height and width must both be divisible by (2^5)', default="(128, 96)")
    parser.add_argument('--image_channels', default=1)
    parser.add_argument('--epochs', help='Number of epochs to train for', default=500000)
    parser.add_argument('--save_interval', help='How many epochs to go between saves/outputs', default=100)
    parser.add_argument('--output_directory', help="Directoy to save weights and images to.", default="./data/output/test")

    args = parser.parse_args()

    dcgan = DCGAN(args.load_discriminator, args.load_generator, args.output_directory, literal_eval(args.image_size),
                  args.image_channels, args.builtin_dataset)
    if args.sample == -1:
        dcgan.train(epochs=int(args.epochs), image_path=args.data, batch_size=int(args.batch_size), save_interval=int(args.save_interval))
    else:
        dcgan.generate_imgs(int(args.sample), literal_eval(args.sample_thresholds), "")
