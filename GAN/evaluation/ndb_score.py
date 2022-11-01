import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np


def process(images):
    images = tf.reduce_mean(images, axis=3)
    images = tf.reshape(images, (tf.shape(images)[0], -1))
    return images

def compute_ndb_score(training_data_classes,generated_data_classes,num_classes,z_threshold):
    ndb = []
    NT = len(training_data_classes)
    NG = len(generated_data_classes)
    for i in range(num_classes):
        nt = np.sum(training_data_classes==i)
        pt = nt/len(training_data_classes) #training data proportion for bin
        ng = np.sum(generated_data_classes==i)
        pg = ng/len(generated_data_classes) #generated data proportion for bin
        P = (nt+ng)/(NT+NG)
        SE = (P*(1-P)*((1/NT)+(1/NG)))**0.5
        if abs((pt-pg)/SE) > z_threshold:
            ndb.append(i)
    return len(ndb)/num_classes

class GAN(tf.keras.Model):
    """
    Accepts the generator,discriminator networks, the latent dimensions and the
    kmeans class object trained on training data
    """
    def __init__(self, discriminator, generator, latent_dim, kmeans_obj=None,num_classes=None,zscore=None):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.kmeans_obj = kmeans_obj
        self.num_classes = num_classes
        self.zscore = zscore
        self.ndb = None
        self.ndb_tracker = None
        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None
        self.dloss = None
        self.gloss = None

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.dloss = tf.keras.metrics.Mean(name="discriminator_loss")
        self.gloss = tf.keras.metrics.Mean(name="generator_loss")
        if self.kmeans_obj:
            self.ndb_tracker = tf.keras.metrics.Mean(name='NDB score')

    @property
    def metrics(self):
        return [self.dloss, self.gloss]

    def ndb_score(self,real_images,fake_images):
        real = process(real_images)
        fake = process(fake_images)
        train_classes = self.kmeans_obj.fit_predict(real)
        generated_classes = self.kmeans_obj.fit_predict(fake)
        return compute_ndb_score(train_classes,generated_classes,self.num_classes,self.zscore)

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(noise)
        if self.kmeans_obj:
            ndb = self.ndb_score(real_images,generated_images)
            self.ndb_tracker.update_state(ndb)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            dloss = self.loss_fn(labels, predictions)
        grads = tape.gradient(dloss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        noise = tf.random.normal(shape=(2 * batch_size, self.latent_dim))
        labels = tf.zeros((2 * batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(noise))
            gloss = self.loss_fn(labels, predictions)
        grads = tape.gradient(gloss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        self.dloss.update_state(dloss)
        self.gloss.update_state(gloss)
        return {"d_loss": self.dloss.result(), "g_loss": self.gloss.result(),"ndb_score": self.ndb_tracker}