import numpy as np
np.random.seed(42)
import os
import tensorflow as tf
tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

class ImageUtil:
    def __init__(self) -> None:
        #self.vgg_model = self.__load_vgg_model()
        pass

    def get_path_list_from_root(root_directory):
        image_paths = []
        # Recursively traverse the directory and its subdirectories
        for root, _, files in os.walk(root_directory):
            files = files[:500]
            for file in files:
                if((file.lower().endswith('.jpg')) or (file.lower().endswith('.jpeg')) or (file.lower().endswith('.png'))):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def load_image(img_path, img_height,img_width):
        raw_inputs = load_img(img_path, target_size=(img_height, img_width))
        return img_to_array(raw_inputs)

    def load_data(root_dir,img_height,img_width):
        image_paths = ImageUtil.get_path_list_from_root(root_directory=root_dir)
        all_images = []
        for img in image_paths:
            #Convert the PIL image from (width, height, channel formant) to a numpy array((height, width, channel))
            all_images.append(ImageUtil.load_image(img,img_height,img_width))
        return all_images
    
    def preprocess_images(images):
        # Normalize pixel values to the range [-1, 1]
        images  = (images.astype(np.float32) - 127.5) / 127.5
        return np.expand_dims(images, axis=-1)
    
    # Function to generate and display images
    def generate_images(generator, latent_dim, num_images=3):
        # generate_images(generator, latent_dim)
        noise=np.random.normal(size=[10,latent_dim])
        gen_image = generator.predict(noise)
        gen_image = 0.5 * gen_image + 0.5
        fig,axe=plt.subplots(2,5)
        fig.suptitle('Generated Images from Noise using GANs')
        idx=0
        for i in range(2):
            for j in range(5):
                axe[i,j].imshow(gen_image[idx])
                idx+=1