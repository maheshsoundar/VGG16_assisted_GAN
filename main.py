import tensorflow as tf
tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
import numpy as np
np.random.seed(42)
from tensorflow.keras.models import Sequential
from keras.applications import vgg16
from tensorflow.keras.optimizers import Adam

from generator import *
from discriminator import *
from utils import *

# Combined model (cGAN)
def build_cgan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

if __name__ == "__main__":
    #Get data and preprocess
    IMG_SIZE = 64 # Choose bigger or smaller according to need
    img_dir = os.path.join(os.getcwd(),'data','happy') #All the images should be present inside data/happy
    images = np.array(ImageUtil.load_data(img_dir,IMG_SIZE,IMG_SIZE))
    images = ImageUtil.preprocess_images(images)

    # Define hyperparameters
    latent_dim = 100
    img_shape = (IMG_SIZE, IMG_SIZE, 3) 

    # Build and compile models
    # Using pretrained model like VGG16 helps discriminator distinguish between fake and real images faster, saving time and computation
    vgg16 = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=img_shape)

    #Build generator, discriminator and combined model
    generator = Generator().build_model(latent_dim, img_shape)
    discriminator = Discriminator().build_model(img_shape,vgg16)
    cgan = build_cgan(generator, discriminator)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.5), metrics=['accuracy'])
    cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.5))

    # Training params
    epochs = 50
    batch_size = 64 #Choose bigger batch size to get better result but takes more time and computation

    for epoch in range(epochs):
        #Get sample of real images and its lables set to 1
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_images,y_real = images[idx],np.ones((batch_size, 1))

        #Get fake samples (noise)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs,y_noise = generator.predict(noise),np.zeros((batch_size, 1))


        # Train the discriminator with both real and fake anc calc avg loss
        d_loss_real = discriminator.train_on_batch(real_images, y_real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, y_noise)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator by running the combined model with discriminator not trainable
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = cgan.train_on_batch(noise, valid_labels)

        # Print progress
        if epoch % 10 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

    #Plot outputs of generator trained
    ImageUtil.generate_images(generator, latent_dim)