from channelexplorer import ChannelExplorer_TF as Cexp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# MODEL, DATASET = 'inceptionv3', 'imagenet'
# MODEL, DATASET = 'vgg16', 'imagenet'

# MODEL, DATASET = 'inceptionv3', 'imagenette'
# MODEL, DATASET = 'vgg16', 'imagenette'

# MODEL, DATASET = 'simple_cnn', 'mnist'

# MODEL, DATASET = 'expression', 'fer2023'

# MODEL, DATASET = 'vgg16', 'eval1'
# MODEL, DATASET = 'inceptionv3', 'eval2'
# MODEL, DATASET = 'vgg16', 'eval2'

MODEL, DATASET = 'GPT2', 'wikitext-2'

# for InceptionV3
# layers_to_show = [
#     'input_1', 'conv2d', 'conv2d_2', 'conv2d_4', 'mixed0', 'mixed1', 'mixed2', 'mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9', 'conv2d_85', 'conv2d_88', 'conv2d_87', 'mixed10', 'predictions'
# ]
# layers_to_show = [
#     'input_1', 'conv2d', 'conv2d_4', 'mixed1', 'mixed3', 'mixed5', 'mixed7', 'mixed9', 'conv2d_85', 'conv2d_88', 'conv2d_87', 'mixed10', 'predictions'
# ]
# layers_to_show = [
#     'input_1', 'mixed6', 'conv2d_60', 'conv2d_63', 'mixed10', 'predictions'
# ]
layers_to_show = 'all'

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
elif MODEL == 'GPT2':
    gpt2_model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # Set pad token
    model = gpt2_model
else:
    raise ValueError(f"Model {MODEL} not supported")

# Remove the model.compile() call for GPT-2
if MODEL != 'GPT2':
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
    # labels = list(map(lambda l: wn.synset_from_pos_and_offset(
    #         l[0], int(l[1:])).name(), info.features['label'].names))
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
    # labels = names=list(map(lambda l: wn.synset_from_pos_and_offset(
    #         l[0], int(l[1:])).name(), info.features['label'].names))
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
elif DATASET == 'wikitext-2':
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    ds = raw_dataset['train'].map(lambda examples: {'text': examples['text']})
    info = None

dataset = ds

if MODEL == 'vgg16':
    vgg16_input_shape = tf.keras.applications.vgg16.VGG16().input.shape[1:3].as_list()
    @tf.function
    def preprocess(x, y):
        x = tf.image.resize(x, vgg16_input_shape, method=tf.image.ResizeMethod.BILINEAR)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        return x, y

elif MODEL == 'inceptionv3':
    inception_input_shape = tf.keras.applications.inception_v3.InceptionV3().input.shape[1:3].as_list()
    @tf.function
    def preprocess(x, y):
        x = tf.image.resize(x, inception_input_shape, method=tf.image.ResizeMethod.BILINEAR)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        return x, y

elif MODEL == 'simple_cnn' or MODEL == 'expression':
    inception_input_shape = tf.keras.applications.inception_v3.InceptionV3().input.shape[1:3].as_list()
    @tf.function
    def preprocess(x, y):
        x = utils.preprocess(x, y, size=inception_input_shape)
        return x, y

elif MODEL == 'GPT2':
    @tf.function
    def preprocess(x):
        encoded = gpt2_tokenizer(x['text'], return_tensors='tf', padding='max_length', max_length=128, truncation=True)
        return encoded['input_ids']

# take 5 samples from the dataset
inputs = []
if MODEL == 'GPT2':
    #     Call arguments received by layer 'transformer' (type TFGPT2MainLayer):
    #   • input_ids=tf.Tensor(shape=(1, 0), dtype=float32)
    #   • past_key_values=tf.Tensor(shape=(1, 128), dtype=int32)
    #   • attention_mask=tf.Tensor(shape=(1, 0), dtype=float32)
    #   • token_type_ids=tf.Tensor(shape=(1, 128), dtype=int32)
    #   • position_ids=tf.Tensor(shape=(1, 128), dtype=int32)
    #   • head_mask=None
    #   • inputs_embeds=None
    #   • encoder_hidden_states=None
    #   • encoder_attention_mask=None
    #   • use_cache=True
    #   • output_attentions=False
    #   • output_hidden_states=True
    #   • return_dict=True
    #   • training=False
    for item in ds.take(5):
        x = item['text']
        inputs.append(x)
    inputs = [input.numpy() for input in inputs]
else:
    for x, y in dataset.take(5):
        x, y = preprocess(x, y)
        inputs.append(x)
    inputs = np.array(inputs)

host = "0.0.0.0"
port = 8000
log_level = "info"

server = Cexp(
    model=model,
    inputs=inputs,
    log_level=log_level,
    layers_to_show=layers_to_show
)

server.run(host=host, port=port)