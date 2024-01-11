from activation_pathway_analysis_backend.apalysis_tf import APAnalysisTensorflowModel
import numpy as np
from nltk.corpus import wordnet as wn
import tensorflow as tf
import tensorflow_datasets as tfds

# MODEL, DATASET = 'inceptionv3', 'imagenet'
MODEL, DATASET = 'vgg16', 'imagenet'

# MODEL, DATASET = 'inceptionv3', 'imagenette'
# MODEL, DATASET = 'vgg16', 'imagenette'

# MODEL, DATASET = 'simple_cnn', 'mnist'

# MODEL, DATASET = 'expression', 'fer2023'

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
        data_dir='/run/media/insane/SSD Games/Tensorflow/tensorflow_datasets'
    )
    labels = list(map(lambda l: wn.synset_from_pos_and_offset(
            l[0], int(l[1:])).name(), info.features['label'].names))
    ds = ds['train']
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
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                     "dimension [1, height, width, channel] or [height, width, channel]")
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

host = "localhost"
port = 8000
log_level = "info"

server = APAnalysisTensorflowModel(
    model=model,
    dataset=dataset,
    label_names=labels,
    preprocess=preprocess,
    preprocess_inverse=preprocess_inv,
    log_level=log_level,
)

server.run_server(host=host, port=port)