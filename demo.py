# Import all libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from activation_pathway_analysis_backend import run_APanalysis_webserver

# To run Tensorflow on CPU or GPU
with_gpu = False
if not with_gpu:
    tf.config.experimental.set_visible_devices([], "GPU")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is Enabled")
else:
    print("GPU is not Enabled")

# Load the model
## Inception V3
# model = tf.keras.applications.inception_v3.InceptionV3(
#     weights='imagenet'
# )

## Resnet 50
# model = tf.keras.applications.resnet50.ResNet50(
#     weights='imagenet'
# )

## VGG 16
model = tf.keras.applications.vgg16.VGG16(
    weights='imagenet'
)

model.compile(loss="categorical_crossentropy", optimizer="adam")

# Load dataset
ds, info = tfds.load(
    'imagenet2012', 
    shuffle_files=False, 
    with_info=True,
    as_supervised=True,
    batch_size=None,
    data_dir='/run/media/insane/My 4TB 2/Big Data/tensorflow_datasets'
)

def preprocess_image(image, label):
    image = tf.image.resize(image, model.input_shape[1:3])
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label

ds = ds.map(preprocess_image)

# Run the web server
run_APanalysis_webserver(
    model=model,
    dataset=ds,
    dataset_info=info,
    host="localhost",
    port=8000,
    log_level="info"
)