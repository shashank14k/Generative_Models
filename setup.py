from setuptools import setup,find_packages

setup(
    name='Generative_Models',
    version='0.0.1',
    packages=find_packages(exclude=['PCA_EigenFace']),
    url='https://github.com/shashank14k/Generative_Models',
    license='',
    author='Shashank',
    author_email='shashank14k@gmail.com',
    description='Class to build VAE model from a list of model parameters and train it'
)
