import tensorflow as tf


# Brining in tensorflow datasets for fashion mnist 
import tensorflow_datasets as tfds
# Bringing in matplotlib for viz stuff
from matplotlib import pyplot as plt


# Use the tensorflow datasets api to bring in the data source
ds = tfds.load('fashion_mnist', split='train')

ds.as_numpy_iterator().next()

# Viz Data and Buiild Dataset

import numpy as np

# Setup connection aka iterator
dataiterator = ds.as_numpy_iterator()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    # Grab an image and label
    sample = dataiterator.next()
    # Plot the image using a specific subplot 
    ax[idx].imshow(np.squeeze(sample['image']))
    # Appending the image label as the plot title 
    ax[idx].title.set_text(sample['label'])

