from functools import lru_cache
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
from fastapi import FastAPI, File, Response
from fastapi.middleware.cors import CORSMiddleware
    
# summary_fn = lambda x: np.percentile(np.abs(x), 90, axis=range(len(x.shape)-1))
summary_fn = lambda x: np.linalg.norm(x, axis=tuple(range(1, len(x.shape)-1)), ord=2)

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

    return nx.node_link_data(simple_activation_pathway_full)


@app.get("/api/labels")
async def read_labels():
    if hasattr(app, 'labels') and app.labels is not None:
        return app.labels.names
    return list(app.dataset_info.features['label'].names)


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

def activation_distance(activation1_summary: dict[str,np.ndarray], activation2_summary: dict[str,np.ndarray]):
    dist = 0
    for act1, act2 in zip(activation1_summary.values(), activation2_summary.values()):
        dist += single_activation_distance(act1, act2)
    return dist

@app.get("/api/analysis/layer/{layer_name}/embedding")
async def analysisLayerEmbedding(layer_name: str):
    global activationsSummary
    this_activation = [activation[layer_name] for activation in activationsSummary]
    act_dist_mat = np.zeros((len(this_activation), len(this_activation)))

    for i, acti in tqdm(enumerate(this_activation), total=len(this_activation)):
        for j, actj in enumerate(this_activation):
            if i == j:
                act_dist_mat[i, j] = 0
                continue
            if i > j:
                continue
            act_dist_mat[i, j] = single_activation_distance(acti, actj)
            act_dist_mat[j, i] = act_dist_mat[i, j]

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(act_dist_mat)
    coords = results.embedding_
    
    return coords.tolist()

@app.get("/api/analysis/layer/{layer_name}/heatmap")
async def analysisLayerHeatmap(layer_name: str):
    global activationsSummary
    layer_activation = [activation[layer_name].tolist() for activation in activationsSummary]
    return layer_activation


@app.post("/api/analysis")
async def analysis(labels: list[int], examplePerClass: int = 50):
    global datasetImgs
    global activations
    global activationsSummary
    global datasetLabels
    global selectedLabels

    selectedLabels = labels

    layers = list(filter(lambda l: 'conv' in l or 'mixed' in l, map(lambda l: l.name, model.layers)))

    datasetImgs = [[] for _ in range(len(labels))]
    activations = [[] for _ in range(len(labels))]
    activationsSummary = [[] for _ in range(len(labels))]
    datasetLabels = [[] for _ in range(len(labels))]

    for img, label in tqdm(app.dataset.filter(
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
        activation = keract.get_activations(model, img, layer_names=layers, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)

        datasetImgs[label_idx].append(img.numpy())
        activations[label_idx].append(activation)
        activationsSummary[label_idx].append({ k: summary_fn(v)[0] for k, v in activation.items() })
        datasetLabels[label_idx].append(label.numpy()[0].item())
        
        if all((len(dtImgs) >= examplePerClass) for dtImgs in datasetImgs):
            break
        
    datasetImgs = [j for i in datasetImgs for j in i]
    activations = [j for i in activations for j in i]
    activationsSummary = [j for i in activationsSummary for j in i]
    datasetLabels = [j for i in datasetLabels for j in i]

    return {
        "selectedClasses": selectedLabels,
        "examplePerClass": len(datasetImgs) // len(selectedLabels),
    }
    
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
    image = (datasetImgs[index][0]*255).astype(np.uint8)
    img = Image.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(content, headers=headers, media_type='image/png')

@app.get("/api/loaded_analysis")
async def loadedAnalysis():
    global datasetImgs
    global datasetLabels
    global selectedLabels
    
    if not selectedLabels:
        return {
            "selectedClasses": [],
            "examplePerClass": 0,
        }

    return {
        "selectedClasses": selectedLabels,
        "examplePerClass": len(datasetImgs) // len(selectedLabels),
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
        
    # MODEL = 'inceptionv3'
    MODEL = 'vgg16'
    # DATASET = 'imagenet'
    DATASET = 'imagenette'


    # Load a demo model
    if MODEL == 'vgg16':
        model = tf.keras.applications.vgg16.VGG16(
            weights='imagenet'
        )
    elif MODEL == 'inceptionv3':
        model = tf.keras.applications.inception_v3.InceptionV3(
            weights='imagenet'
        )
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
            data_dir='/run/media/insane/My 4TB 2/Big Data/tensorflow_datasets'
        )
        labels = tfds.features.ClassLabel(
            names=list(map(lambda l: wn.synset_from_pos_and_offset(
                l[0], int(l[1:])).name(), info.features['label'].names))
        )
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
    
    # Setting the model and dataset
    app.model = model
    app.dataset = ds['train']
    app.dataset_info = info
    app.labels = labels

    host = "localhost"
    port = 8000
    log_level = "info"

    # Starting the server
    # app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parents[0].joinpath('static').resolve(), html=True), name="react_build")
    uvicorn.run(app, host=host, port=port, log_level=log_level)