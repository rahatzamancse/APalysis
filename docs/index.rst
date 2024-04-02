Welcome to ChannelExplorer's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Quickstart (w/ Docker)
----------------------

You can run the following the start the whole dockerized application
quickly in localhost.

.. code:: bash

   docker run -p 8000:8000/tcp channelexplorer

And open ``http://localhost:8000`` in your browser.

Installation
------------

The project is available on PYPI. **Currently the project only supports
python 3.10**. You can install using

.. code:: bash

   pip install apalysis

You will need a running a redis server at the default port for the
project to work. You can install redis using

.. code:: bash

   # Install redis
   sudo apt install redis-server # For debian-based distros
   # sudo pacman -S redis # For arch

   # Run redis
   redis-server --daemonize yes

You can also install redis using docker. You can find the docker image
`here <https://hub.docker.com/_/redis>`__.

Usage
-----

Tensorflow
~~~~~~~~~~

.. code:: python

   # Import everything
   from channelexplorer import ChannelExplorer_TF as Cexp
   import tensorflow as tf
   import tensorflow_datasets as tfds
   import numpy as np
   from nltk.corpus import wordnet as wn

   # Load the tensorflow model
   model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
   model.compile(loss="categorical_crossentropy", optimizer="adam")

   # Load the dataset
   # The dataset needs to be in tensorflow_datasets format
   ds, info = tfds.load(
       'imagenette/320px-v2',
       shuffle_files=False,
       with_info=True,
       as_supervised=True,
       batch_size=None,
   )
   # For classification, the labels is just a list of names for each encoded class
   labels = names=list(map(lambda l: wn.synset_from_pos_and_offset(
           l[0], int(l[1:])).name(), info.features['label'].names))
   dataset = ds['train']

   # To feed the dataset into the model, we need to provide a preprocessing function
   vgg16_input_shape = tf.keras.applications.vgg16.VGG16().input.shape[1:3].as_list()
   @tf.function
   def preprocess(x, y):
       x = tf.image.resize(x, vgg16_input_shape, method=tf.image.ResizeMethod.BILINEAR)
       x = tf.keras.applications.vgg16.preprocess_input(x)
       return x, y

   # We also need to provide a preprocessing inverse function to convert the preprocessed image back to the original image
   # This is used to display the original image in the frontend
   # This will be automated in the future
   def preprocess_inv(x, y):
       x = x.squeeze(0)
       # Again adding the mean pixel values
       x[:, :, 0] += 103.939
       x[:, :, 1] += 116.779
       x[:, :, 2] += 123.68
       x = x[:, :, ::-1]
       x = np.clip(x, 0, 255).astype('uint8')
       return x, y

   # Create the server and run it
   server = Cexp(
       model=model,
       dataset=dataset,
       label_names=labels,
       preprocess=preprocess,
       preprocess_inverse=preprocess_inv,
       log_level="info",
   )
   server.run_server(host="localhost", port=8000)

PyTorch
~~~~~~~

Very similar to the above example.

.. code:: python

   # Imports
   from channelexplorer import ChannelExplorer_Torch as Cexp
   import torch
   import torchvision.models as models
   import torchvision.datasets as datasets
   import torchvision.transforms as transforms
   from torch.utils.data import DataLoader
   import numpy as np
   from nltk.corpus import wordnet as wn

   # Load the model
   model = models.vgg16(pretrained=True)
   loss_fn = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters())

   # We do not need any preprocessing function to provide to the server for pytorch. we can provide it to the dataset instead.
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
   ])
   dataset = datasets.MNIST('/run/media/insane/SSD Games/Pytorch', train=True, download=True, transform=transform)

   # Start the server
   server = Cexp(
       model=model,
       input_shape=(1, 3, 224, 224),
       dataset=dataset,
       label_names=labels,
       log_level="info",
   )

   server.run_server(host="localhost", port=8000)
