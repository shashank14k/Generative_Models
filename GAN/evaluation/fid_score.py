import tensorflow as tf
import numpy as np
from scipy import linalg

class FidScore:
    def __init__(self, real_images, generated_images, iterations,
                 model=tf.keras.applications.InceptionV3(include_top=False,
                                                         weights="imagenet",
                                                         pooling='avg')):
        self.real_images = real_images
        self.generated_images = generated_images
        self.embeddings_model = model
        self.real_embeddings = self.create_embeddings(real_images, iterations)
        self.generated_embeddings = self.create_embeddings(generated_images,
                                                           iterations)
        return self.computed_fid()

    def create_embeddings(self, dataloader, iters):
        embeddings = []
        for _ in range(iters):
            images = next(iter(dataloader))
            embeddings.extend(self.embeddings_model.predict(images))

        return np.array(embeddings)

    def computed_fid(self):
        mu1, sigma1 = self.real_embeddings.mean(axis=0), np.cov(self.real_embeddings,
                                                           rowvar=False)
        mu2, sigma2 = self.generated_embeddings.mean(axis=0), np.cov(
            self.generated_embeddings, rowvar=False)

        diff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = linalg.sqrtm(sigma1.dot(sigma2))

        fid = diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
