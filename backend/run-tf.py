from channelexplorer import ChannelExplorer_TF as Cexp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers

# MODEL, DATASET = ['vgg16', 'inceptionv3', 'GPT2'], ['imagenette', 'imagenette', 'GPT2-custom']
# MODEL, DATASET = ['vgg16', 'GPT2'], ['imagenette', 'GPT2-custom']
# MODEL, DATASET = ['vgg16', 'inceptionv3'], ['imagenette', 'imagenette']
# MODEL, DATASET = ['vgg16'], ['imagenette']
MODEL, DATASET = ['vit', 'inceptionv3', 'vgg16', 'GPT2'], ['imagenette', 'imagenette', 'imagenette', 'GPT2-custom']
# MODEL, DATASET = ['clip'], ['imagenette']

inception_layers_to_show = [
    'input_1', 'conv2d', 'conv2d_2', 'conv2d_4', 'mixed0', 'mixed1', 'mixed2', 'mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9', 'conv2d_85', 'conv2d_88', 'conv2d_87', 'mixed10', 'predictions'
]

layers_to_show = ['all', inception_layers_to_show, 'all', 'all']
# layers_to_show = ['all', 'all']
# layers_to_show = ['all', 'all']
# layers_to_show = ['all', inception_layers_to_show]

# Load a demo model
models = []
for model_name in MODEL:
    if model_name == 'vgg16':
        model = tf.keras.applications.vgg16.VGG16(
            weights='imagenet'
        )
        models.append(model)
    elif model_name == 'inceptionv3':
        model = tf.keras.applications.inception_v3.InceptionV3(
            weights='imagenet'
        )
        models.append(model)
    elif model_name == 'simple_cnn':
        model = tf.keras.models.load_model('../analysis/saved_model/keras_mnist_cnn')
        models.append(model)
    elif model_name == 'GPT2':
        model = transformers.TFAutoModelForCausalLM.from_pretrained('gpt2')
        model.config.pad_token_id = model.config.eos_token_id
        models.append(model)
    elif model_name == 'vit':
        model = transformers.TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        models.append(model)
    
if len(models) == 0:
    raise ValueError(f"Model {MODEL} not supported")

# Remove the model.compile() call for GPT-2
for model_name, model in zip(MODEL, models):
    if model_name not in ['GPT2', 'vit']:
        model.compile(loss="categorical_crossentropy", optimizer="adam")

# Load dataset
datasets = []
for dataset_name in DATASET:
    if dataset_name == 'imagenet':
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
        datasets.append(ds)
    if dataset_name == 'imagenette':
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
        datasets.append(ds)
    if dataset_name == 'mnist':
        ds, info = tfds.load(
            'mnist',
            shuffle_files=False,
            with_info=True,
            as_supervised=True,
            batch_size=None,
        )
        labels = list(map(str, range(10)))
        ds = ds['train']
        datasets.append(ds)
    if dataset_name == 'fer2023':
        ds = tf.keras.utils.image_dataset_from_directory(
            '/home/insane/u/AffectiveTDA/FER-2013/train',
            seed=123,
            image_size=(48, 48),
            color_mode='grayscale',
            batch_size=None,
        )
        labels = ds.class_names
        datasets.append(ds)
    if dataset_name == 'GPT2-custom':
        dataset = [
            'Hello, how are you?',
            'I am fine, thank you!',
            'What is your name?',
            'My name is John.',
            'How old are you?',
            'I am 20 years old.',
            'What is your favorite color?',
            'My favorite color is blue.',
            'What is your favorite food?',
        ]
        datasets.append(dataset)

processors = []
for model_name, model in zip(MODEL, models):
    if model_name == 'vgg16':
        vgg16_input_shape = model.input.shape[1:3].as_list()
        @tf.function
        def preprocess(x, y):
            x = tf.image.resize(x, vgg16_input_shape, method=tf.image.ResizeMethod.BILINEAR)
            x = tf.keras.applications.vgg16.preprocess_input(x)
            x = tf.expand_dims(x, axis=0)
            return x, y
        processors.append(preprocess)
    elif model_name == 'inceptionv3':
        inception_input_shape = tf.keras.applications.inception_v3.InceptionV3().input.shape[1:3].as_list()
        @tf.function
        def preprocess(x, y):
            x = tf.image.resize(x, inception_input_shape, method=tf.image.ResizeMethod.BILINEAR)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            x = tf.expand_dims(x, axis=0)
            return x, y
        processors.append(preprocess)
    elif model_name == 'simple_cnn':
        inception_input_shape = tf.keras.applications.inception_v3.InceptionV3().input.shape[1:3].as_list()
        @tf.function
        def preprocess(x, y):
            x = utils.preprocess(x, y, size=inception_input_shape)
            return x, y
        processors.append(preprocess)
    elif model_name == 'GPT2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        def preprocess(x):
            return [tokenizer(i, return_tensors='tf') for i in x]
        processors.append(preprocess)
    elif model_name == 'vit':
        def preprocess(x, y):
            feature_extractor = transformers.ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            x = tf.image.resize(x, (224, 224)) / 255.0
            x = feature_extractor(images=x, return_tensors="tf")
            x = x['pixel_values']
            return x, y
        processors.append(preprocess)

# take 5 samples from the dataset
all_inputs = []
for model_name, dataset, processor in zip(MODEL, datasets, processors):
    inputs = []
    if model_name == 'GPT2':
        inputs = preprocess(dataset)
    else:
        for x, y in dataset.take(5):
            x, y = processor(x, y)
            inputs.append(x)
        inputs = np.concatenate(inputs, axis=0)
    all_inputs.append(inputs)

host = "0.0.0.0"
port = 8000
log_level = "info"

server = Cexp(
    models=models,
    all_inputs=all_inputs,
    log_level=log_level,
    layers_to_show=layers_to_show
)

server.run(host=host, port=port)