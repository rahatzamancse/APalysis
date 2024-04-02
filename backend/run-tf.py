from channelexplorer import ChannelExplorer_TF as Cexp
import numpy as np
from nltk.corpus import wordnet as wn
import tensorflow as tf
import tensorflow_datasets as tfds

MODEL, DATASET = 'inceptionv3', 'imagenet'
# MODEL, DATASET = 'vgg16', 'imagenet'

# MODEL, DATASET = 'inceptionv3', 'imagenette'
# MODEL, DATASET = 'vgg16', 'imagenette'

# MODEL, DATASET = 'simple_cnn', 'mnist'

# MODEL, DATASET = 'expression', 'fer2023'

# MODEL, DATASET = 'vgg16', 'eval1'
# MODEL, DATASET = 'inceptionv3', 'eval2'
# MODEL, DATASET = 'vgg16', 'eval2'

# for InceptionV3
# layers_to_show = [
#     'input_1', 'conv2d', 'conv2d_2', 'conv2d_4', 'mixed0', 'mixed1', 'mixed2', 'mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9', 'conv2d_85', 'conv2d_88', 'conv2d_87', 'mixed10', 'predictions'
# ]
layers_to_show = [
    'input_1', 'mixed6', 'conv2d_60', 'conv2d_63', 'mixed10', 'predictions'
]
# layers_to_show = 'all'

# Load a demo model
if MODEL == 'vgg16':
    model = tf.keras.applications.vgg16.VGG16(
        weights='imagenet'
    )
elif MODEL == 'inceptionv3':
    model = tf.keras.applications.inception_v3.InceptionV3(
        weights='imagenet'
    )
elif MODEL == 'simple_cnn':
    model = tf.keras.models.load_model(
        '../analysis/saved_model/keras_mnist_cnn')
elif MODEL == 'expression':
    model = tf.keras.models.load_model(
        '/home/insane/u/AffectiveTDA/fer_model')
else:
    raise ValueError(f"Model {MODEL} not supported")

model.compile(loss="categorical_crossentropy", optimizer="adam")

# Load dataset
if DATASET == 'imagenet':
    ds, info = tfds.load(
        'imagenet2012',
        shuffle_files=False,
        with_info=True,
        as_supervised=True,
        batch_size=None,
        data_dir='/run/media/insane/Games/Tensorflow/tensorflow_datasets'
    )
    labels = list(map(lambda l: wn.synset_from_pos_and_offset(
            l[0], int(l[1:])).name(), info.features['label'].names))
    ds = ds['train']
elif DATASET.startswith('eval'):
    # create dataset from directory
    ds = tf.keras.utils.image_dataset_from_directory(
        f'/home/insane/u/apalysis-evaluation/dataset-{DATASET[4:]}',
        seed=123,
        image_size=(224, 224),
        batch_size=None,
        labels='inferred',
        label_mode='int',
    )
    info = None
    labels = ['white shark', 'tiger shark', 'african dog', 'persian cat', 'egyptian cat']
elif DATASET == 'imagenette':
    ds, info = tfds.load(
        'imagenette/320px-v2',
        shuffle_files=False,
        with_info=True,
        as_supervised=True,
        batch_size=None,
    )
    labels = names=list(map(lambda l: wn.synset_from_pos_and_offset(
            l[0], int(l[1:])).name(), info.features['label'].names))
    ds = ds['train']
elif DATASET == 'mnist':
    ds, info = tfds.load(
        'mnist',
        shuffle_files=False,
        with_info=True,
        as_supervised=True,
        batch_size=None,
    )
    labels = list(map(str, range(10)))
    ds = ds['train']
elif DATASET == 'fer2023':
    ds = tf.keras.utils.image_dataset_from_directory(
        '/home/insane/u/AffectiveTDA/FER-2013/train',
        seed=123,
        image_size=(48, 48),
        color_mode='grayscale',
        batch_size=None,
    )
    info = None
    labels = ds.class_names

dataset = ds

if MODEL == 'vgg16':
    vgg16_input_shape = tf.keras.applications.vgg16.VGG16().input.shape[1:3].as_list()
    @tf.function
    def preprocess(x, y):
        x = tf.image.resize(x, vgg16_input_shape, method=tf.image.ResizeMethod.BILINEAR)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        return x, y

    def preprocess_inv(x, y):
        x = x.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of dimension [1, height, width, channel] or [height, width, channel]")
        if len(x.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")

        # perform the inverse of the preprocessing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x, y

elif MODEL == 'inceptionv3':
    inception_input_shape = tf.keras.applications.inception_v3.InceptionV3().input.shape[1:3].as_list()
    @tf.function
    def preprocess(x, y):
        x = tf.image.resize(x, inception_input_shape, method=tf.image.ResizeMethod.BILINEAR)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        return x, y

    def preprocess_inv(x, y):
        x = ((x / 2 + 0.5) * 255).astype(np.uint8).squeeze()
        return x, y

elif MODEL == 'simple_cnn' or MODEL == 'expression':
    inception_input_shape = tf.keras.applications.inception_v3.InceptionV3().input.shape[1:3].as_list()
    @tf.function
    def preprocess(x, y):
        x = utils.preprocess(x, y, size=inception_input_shape)

    def preprocess_inv(x, y):
        x = ((x / 2 + 0.5) * 255).astype(np.uint8).squeeze()
        return x, y

host = "0.0.0.0"
port = 8000
log_level = "info"

server = Cexp(
    model=model,
    dataset=dataset,
    label_names=labels,
    preprocess=preprocess,
    preprocess_inverse=preprocess_inv,
    log_level=log_level,
    layers_to_show=layers_to_show
)

server.run(host=host, port=port)