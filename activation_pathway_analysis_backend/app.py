from typing import Literal
from pydantic import BaseModel
import tensorflow as tf

# from .utils import get_model_layout, model_to_graph, pred_to_name, preprocess, remove_intermediate_node
import utils
import networkx as nx
import numpy as np
from PIL import Image
import keract
import io
from tqdm import tqdm
from sklearn import manifold

import uvicorn
from fastapi import FastAPI, File, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import normalize
    
# summary_fn_image = lambda x: np.percentile(np.abs(x), 90, axis=range(len(x.shape)-1))
summary_fn_image = lambda x: np.linalg.norm(x, axis=tuple(range(1, len(x.shape)-1)), ord=2)
summary_fn_dense = lambda x: x

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

datasetImgs = []
datasetLabels = []
activations = []
activationsSummary = []
selectedLabels = []
predictions = []
shuffled = False

feature_hunt_activated_channels = None

@app.get("/api/model/")
async def read_model():
    activation_pathway_full = utils.model_to_graph(app.model)
    simple_activation_pathway_full = utils.remove_intermediate_node(activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['layer_type'] not in ['Conv2D', 'Dense', 'InputLayer', 'Concatenate'])

    node_pos = utils.get_model_layout(simple_activation_pathway_full)

    # normalize node positions to be between 0 and 1
    min_x = min(pos[0] for pos in node_pos.values())
    max_x = max(pos[0] for pos in node_pos.values())
    min_y = min(pos[1] for pos in node_pos.values())
    max_y = max(pos[1] for pos in node_pos.values())

    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1

    node_pos = {
        node: ((pos[0] - min_x)/(max_x - min_x), (pos[1] - min_y)/(max_y - min_y)) for node, pos in node_pos.items()
    }

    for node, pos in node_pos.items():
        simple_activation_pathway_full.nodes[node]['pos'] = { 'x': pos[0], 'y': pos[1] }
    
    return {
        'graph': nx.node_link_data(simple_activation_pathway_full),
        'meta': {
            'depth': max(nx.shortest_path_length(simple_activation_pathway_full, next(n for n, d in simple_activation_pathway_full.nodes(data=True) if d['layer_type'] == 'InputLayer')).values())
        }
    }


@app.get("/api/labels")
async def read_labels():
    if hasattr(app, 'labels') and app.labels is not None:
        return app.labels.names
    return list(app.dataset_info.features['label'].names)

feature_hunt_image = None

@app.get("/api/polygon/getimage")
async def polygon_getimage():
    global feature_hunt_image
    if feature_hunt_image is None:
        return {
            "message": "no image",
        }
    image = ((feature_hunt_image / 2 + 0.5) * 255).astype(np.uint8).squeeze()
    img = Image.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(content, headers=headers, media_type='image/png')

@app.post("/api/polygon/image")
async def polygon(file: bytes = File(...)):
    global feature_hunt_image
    img = Image.open(io.BytesIO(file))
    img = np.array(img)
    feature_hunt_image = utils.preprocess((img,-1), size=app.model.input.shape[1:3].as_list())[0].numpy()
    
    return {
        "message": "success",
    }

class Point(BaseModel):
    x: float
    y: float
    

@app.post("/api/polygon/points")
async def polygon_points(points: list[Point]):
    global feature_hunt_image
    global feature_hunt_activated_channels
    if feature_hunt_image is None:
        return {
            "message": "no image",
        }
        
    # get the activations of the image
    layers = list(map(lambda l: l.name, filter(lambda l: isinstance(l, (
        # tf.keras.layers.InputLayer,
        tf.keras.layers.Conv2D,
        # tf.keras.layers.Dense,
        # tf.keras.layers.Flatten,
        # tf.keras.layers.Concatenate,
    )), app.model.layers)))
    activation = keract.get_activations(app.model, np.array([feature_hunt_image]), layer_names=layers, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
    
    mask_img = utils.get_mask_img([[p.x, p.y] for p in points], feature_hunt_image.shape[0:2])

    feature_hunt_activated_channels = utils.get_mask_activation_channels(mask_img, activation, summary_fn_image)
    
    return {
        "activated_channels": feature_hunt_activated_channels,
    }
    
@app.get("/api/polygon/activated_channels")
async def polygon_activated_channels():
    global feature_hunt_activated_channels
    if feature_hunt_activated_channels is None:
        return {
            "message": "no image",
        }
    return {
        "activated_channels": feature_hunt_activated_channels,
    }

@app.post("/api/analysis")
async def analysis(labels: list[int], examplePerClass: int = 50, shuffle: bool = False):
    global datasetImgs
    global activations
    global activationsSummary
    global datasetLabels
    global selectedLabels
    global shuffled
    global predictions
    
    shuffled = shuffle
    selectedLabels = labels

    layers = list(map(lambda l: l.name, filter(lambda l: isinstance(l, (
        # tf.keras.layers.InputLayer,
        tf.keras.layers.Conv2D,
        tf.keras.layers.Dense,
        tf.keras.layers.Flatten,
        tf.keras.layers.Concatenate,
    )), app.model.layers)))

    datasetImgs = [[] for _ in range(len(labels))]
    activations = [[] for _ in range(len(labels))]
    activationsSummary = [[] for _ in range(len(labels))]
    datasetLabels = [[] for _ in range(len(labels))]
    
    def shuffle_or_noshuffle(dataset, shuffle: bool = False):
        if shuffle:
            print("Shuffling dataset")
            # TODO: make the shuffling for whole data instead of first 1000 data
            return dataset.shuffle(10000)
        else:
            print("Not Shuffling dataset")
            return dataset

    for img, label in tqdm(shuffle_or_noshuffle(
            app.dataset, shuffle=shuffled
        ).filter(
            lambda img, label: tf.reduce_any(tf.equal(label, labels))
        ).map(
            lambda img,label: utils.preprocess((img,label), size=app.model.input.shape[1:3].as_list())
        ).batch(
            1
        ), total=examplePerClass*len(labels)):

        if len(datasetImgs[labels.index(label)]) >= examplePerClass:
            continue
            
        label_idx = labels.index(label)
        
        # Get activations
        activation = keract.get_activations(app.model, img, layer_names=layers, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
        
        datasetImgs[label_idx].append(img.numpy())
        activations[label_idx].append(activation)
        
        activationSummary = {}
        for k, v in activation.items():
            if len(v[0].shape) == 1:
                # dense layer
                activationSummary[k] = summary_fn_dense(v)[0]
            elif len(v[0].shape) == 3:
                # Image layer
                activationSummary[k] = summary_fn_image(v)[0]
        activationsSummary[label_idx].append(activationSummary)

        datasetLabels[label_idx].append(label.numpy()[0].item())
        
        if all((len(dtImgs) >= examplePerClass) for dtImgs in datasetImgs):
            break
        
    datasetImgs = [j for i in datasetImgs for j in i]
    activations = [j for i in activations for j in i]
    activationsSummary = [j for i in activationsSummary for j in i]
    datasetLabels = [j for i in datasetLabels for j in i]
    

    # Get the prediction with argmax
    predictions = []
    for i in range(len(activations)):
        predictions.append(np.argmax(activations[i][layers[-1]][0]).item())
        
    return {
        "selectedClasses": selectedLabels,
        "examplePerClass": len(datasetImgs) // len(selectedLabels),
        "shuffled": shuffled,
        "predictions": predictions,
    }
    
    
@app.get("/api/analysis/layer/{layer}/argmax")
async def get_argmax(layer: str):
    global activations
    return [np.argmax(activation[layer][0]).item() for activation in activations]

@app.get("/api/analysis/image/{image_idx}/layer/{layer_name}/filter/{filter_index}")
async def get_activation_images(image_idx: int, layer_name: str, filter_index: int):
    image = activations[image_idx][layer_name][0, :, :, filter_index]
    image -= image.min()
    image = (image - np.percentile(image, 10)) / (np.percentile(image, 90) - np.percentile(image, 10))
    image = np.clip(image, 0, 1)
    # invert the image
    image = 1 - image
    image = (image * 255).astype(np.uint8)
    image = np.stack((image,)*3, axis=-1)
    img = Image.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(content, headers=headers, media_type='image/png')


def single_activation_distance(activation1_summary: np.ndarray, activation2_summary: np.ndarray):
    return np.sum(np.abs(activation1_summary - activation2_summary))

def single_activation_jaccard_distance(activation1_summary: np.ndarray, activation2_summary: np.ndarray, threshold: float = 0.5):
    assert len(activation1_summary) == len(activation2_summary)
    activation1_summary = activation1_summary > threshold
    activation2_summary = activation2_summary > threshold
    size = len(activation1_summary)
    num_differences = sum(int(activation1_summary[i] != activation2_summary[i]) for i in range(size))
    distance = num_differences / size
    return distance

def activation_distance(activation1_summary: dict[str,np.ndarray], activation2_summary: dict[str,np.ndarray]):
    dist = 0
    for act1, act2 in zip(activation1_summary.values(), activation2_summary.values()):
        dist += single_activation_distance(act1, act2)
    return dist

@app.get("/api/analysis/layer/{layer_name}/embedding")
async def analysisLayerEmbedding(layer_name: str, normalization: Literal['none', 'row', 'col']='none', method: Literal['mds', 'tsne']="mds", distance: Literal['euclidean', 'jaccard']="euclidean"):
    global activationsSummary
    this_activation = [activation[layer_name] for activation in activationsSummary]
    this_activation = np.array(this_activation)

    if normalize == 'row':
        this_activation = normalize(this_activation, axis=1, norm='l1')
    elif normalize == 'col':
        this_activation = normalize(this_activation, axis=0, norm='l1')

    act_dist_mat = np.zeros((len(this_activation), len(this_activation)))

    for i, acti in tqdm(enumerate(this_activation), total=len(this_activation)):
        for j, actj in enumerate(this_activation):
            if i == j:
                act_dist_mat[i, j] = 0
                continue
            if i > j:
                continue
            if distance == 'euclidean':
                act_dist_mat[i, j] = single_activation_distance(acti, actj)
            elif distance == 'jaccard':
                act_dist_mat[i, j] = single_activation_jaccard_distance(acti, actj)

            act_dist_mat[j, i] = act_dist_mat[i, j]

    if method == 'mds':
        embedding_model = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6, normalized_stress='auto')
    elif method == 'tsne':
        embedding_model = manifold.TSNE(n_components=2, metric='precomputed', random_state=6, perplexity=min(30, len(this_activation)-1), init='random')
    coords = embedding_model.fit_transform(act_dist_mat)
    
    return coords.tolist()


@app.get("/api/analysis/layer/{layer_name}/heatmap")
async def analysisLayerHeatmap(layer_name: str):
    global activationsSummary
    layer_activation = [activation[layer_name].tolist() for activation in activationsSummary]
    return layer_activation


@app.get("/api/analysis/allembedding")
async def analysisAllEmbedding():
    global activationsSummary
    
    act_dist_mat = np.zeros((len(activationsSummary), len(activationsSummary)))
    for i, acti in tqdm(enumerate(activationsSummary), total=len(activationsSummary)):
        for j, actj in enumerate(activationsSummary):
            if i == j:
                act_dist_mat[i, j] = 0
                continue
            if i > j:
                continue
            act_dist_mat[i, j] = activation_distance(acti, actj)
            act_dist_mat[j, i] = act_dist_mat[i, j]

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(act_dist_mat)
    coords = results.embedding_

    return coords.tolist()
    
@app.get("/api/analysis/images/{index}")
async def inputImages(index: int):
    global datasetImgs
    if index < 0 or index >= len(datasetImgs):
        return Response(status_code=404)
    image = ((datasetImgs[index][0] / 2 + 0.5) * 255).astype(np.uint8).squeeze()
    # image = (datasetImgs[index][0]*255).astype(np.uint8)
    img = Image.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(content, headers=headers, media_type='image/png')
    
@app.get("/api/analysis/layer/{layer_name}/{channel}/heatmap/{image}")
async def analysisLayerHeatmap(layer_name: str, channel: int, image: int):
    global activations
    global datasetImgs
    
    image = utils.get_activation_overlay(datasetImgs[image][0].squeeze(), activations[image][layer_name][0][:, :, channel], alpha=0.8)

    image = ((image / 2 + 0.5) * 255).astype(np.uint8)
    # image = (datasetImgs[index][0]*255).astype(np.uint8)
    img = Image.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(content, headers=headers, media_type='image/png')

@app.get("/api/analysis/layer/{layer_name}/{channel}/kernel")
async def analysisLayerKernel(layer_name: str, channel: int):
    kernel = app.model.get_layer(layer_name).get_weights()[0][:,:,0,channel]
    kernel = ((kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255).astype(np.uint8)
    img = Image.fromarray(kernel)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(content, headers=headers, media_type='image/png')

@app.get("/api/analysis/predictions")
async def analysisPredictions():
    global predictions
    return predictions

@app.get("/api/loaded_analysis")
async def loadedAnalysis():
    global datasetImgs
    global datasetLabels
    global selectedLabels
    global shuffled
    global predictions
    
    if not selectedLabels:
        return {
            "selectedClasses": [],
            "examplePerClass": 0,
        }

    return {
        "selectedClasses": selectedLabels,
        "examplePerClass": len(datasetImgs) // len(selectedLabels),
        "shuffled": shuffled,
        "predictions": predictions,
    }

if __name__ == '__main__':
    import tensorflow_datasets as tfds
    from nltk.corpus import wordnet as wn

    # # TODO: Remove this
    # To run Tensorflow on CPU or GPU
    with_gpu = False
    if not with_gpu:
        tf.config.experimental.set_visible_devices([], "GPU")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU is Enabled")
    else:
        print("GPU is not Enabled")
        
    # MODEL, DATASET = 'inceptionv3', 'imagenet'
    # MODEL, DATASET = 'vgg16', 'imagenet'

    # MODEL, DATASET = 'inceptionv3', 'imagenette'
    # MODEL, DATASET = 'vgg16', 'imagenette'

    # MODEL, DATASET = 'simple_cnn', 'mnist'
    
    MODEL, DATASET = 'expression', 'fer2023'
    

    # Area of regions activated?
    # Inter-layer relationships like if first layer is 

    # Activation masks. 

    # https://visxai.io/


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
        model = tf.keras.models.load_model('../analysis/saved_model/keras_mnist_cnn')
    elif MODEL == 'expression':
        model = tf.keras.models.load_model('/home/insane/u/AffectiveTDA/fer_model')
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
        labels = tfds.features.ClassLabel(
            names=list(map(lambda l: wn.synset_from_pos_and_offset(
                l[0], int(l[1:])).name(), info.features['label'].names))
        )
        ds = ds['train']
    elif DATASET == 'imagenette':
        ds, info = tfds.load(
            'imagenette/320px-v2', 
            shuffle_files=False, 
            with_info=True,
            as_supervised=True,
            batch_size=None,
        )
        labels = tfds.features.ClassLabel(
            names=list(map(lambda l: wn.synset_from_pos_and_offset(
                l[0], int(l[1:])).name(), info.features['label'].names))
        )
        ds = ds['train']
    elif DATASET == 'mnist':
        ds, info = tfds.load(
            'mnist', 
            shuffle_files=False, 
            with_info=True,
            as_supervised=True,
            batch_size=None,
        )
        labels = tfds.features.ClassLabel(names=list(map(str, range(10))))
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
        labels = tfds.features.ClassLabel(names=ds.class_names)

        
    # Setting the model and dataset
    app.model = model
    app.dataset = ds
    app.dataset_info = info
    app.labels = labels

    host = "localhost"
    port = 8000
    log_level = "info"

    # Starting the server
    # app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parents[0].joinpath('static').resolve(), html=True), name="react_build")
    uvicorn.run(app, host=host, port=port, log_level=log_level)