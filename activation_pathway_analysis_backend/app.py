import tensorflow as tf

# from .utils import get_model_layout, model_to_graph, pred_to_name, preprocess, remove_intermediate_node
import utils
import networkx as nx
import numpy as np
from PIL import Image
from io import BytesIO
import keract
import io
from tqdm import tqdm
from sklearn import manifold

import uvicorn
from fastapi import FastAPI, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
    

summary_fn = lambda x: np.percentile(np.abs(x), 90, axis=range(len(x.shape)-1))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

currentImg = None
currentActivations = None
datasetImgs = []

@app.get("/api/labels")
async def read_labels():
    return list(app.dataset_info.features['label'].names)

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

@app.post("/api/exampleimg/")
async def example_img(file: bytes = File(...)):
    global currentImg
    global currentActivations
    currentImg = np.array(Image.open(BytesIO(file)))
    input_img, _ = utils.preprocess((np.array([currentImg]), -1), size=app.model.input.shape[1:3].as_list())
    currentActivations = keract.get_activations(app.model, input_img, layer_names=None, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)

    pred: tf.Tensor = app.model(input_img)
    pred_label = app.dataset_info.features['label'].names[pred[0].numpy().argmax()]
    return { "prediction": pred_label }

@app.get("/api/activations/{layer_name}")
async def get_activation(layer_name: str):
    global currentActivations

    images = currentActivations[layer_name]
    img_summary = summary_fn(currentActivations[layer_name]).tolist()
    return {
        "n_filters": images.shape[-1],
        "threshold": 0.5,
        "img_summary": img_summary
    }

@app.get("/api/activations/{layer_name}/image/{index}")
async def get_activations(layer_name: str, index: int):
    global currentActivations
    image = currentActivations[layer_name][0, :, :, index]

    image = np.abs(image)
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

@app.post("/api/analysis")
async def analysis(labels: list[int]):
    global datasetImgs

    def single_activation_distance(activation1: np.ndarray, activation2: np.ndarray):
        return np.sum(np.abs(summary_fn(activation1) - summary_fn(activation2)))

    def activation_distance(activation1: dict[str,np.ndarray], activation2: dict[str,np.ndarray]):
        dist = 0
        for act1, act2 in zip(activation1.values(), activation2.values()):
            dist += single_activation_distance(act1, act2)
        return dist

    N = 10
    activations = []
    layers = list(filter(lambda l: 'conv' in l or 'mixed' in l, map(lambda l: l.name, model.layers)))
    img_labels = []

    for img, label in tqdm(app.dataset.filter(
            lambda img,label: tf.reduce_any(tf.equal(label, labels))
        ).map(
            lambda img,label: utils.preprocess((img,label), size=app.model.input.shape[1:3].as_list())
        ).batch(
            1
        ).take(
            N
        ), total=N):
        # Get activations
        datasetImgs.append(img.numpy())
        activation = keract.get_activations(model, img, layer_names=layers, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
        activations.append(activation)
        img_labels.append(label)

    act_dist_mat = np.zeros((len(activations), len(activations)))

    for i, acti in tqdm(enumerate(activations), total=len(activations)):
        for j, actj in enumerate(activations):
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

    return {
        "coords": coords.tolist(),
        "labels": labels
    }

@app.get("/api/analysis/image/{index}")
async def get_analysis_img(index: int):
    global datasetImgs
    img = datasetImgs[index]
    img = np.squeeze(img)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': f'inline; filename="image_{index}.png"'}
    return Response(content, headers=headers, media_type='image/png')


if __name__ == '__main__':

    import tensorflow_datasets as tfds
    # # TODO: Remove this
    # To run Tensorflow on CPU or GPU
    with_gpu = False
    if not with_gpu:
        tf.config.experimental.set_visible_devices([], "GPU")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU is Enabled")
    else:
        print("GPU is not Enabled")


    # Load a demo model
    # model = tf.keras.applications.vgg16.VGG16(
    #     weights='imagenet'
    # )
    model = tf.keras.applications.inception_v3.InceptionV3(
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

    # Setting the model and dataset
    app.model = model
    app.dataset = ds['train']
    app.dataset_info = info

    host = "localhost"
    port = 8000
    log_level = "info"

    # Starting the server
    # app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parents[0].joinpath('static').resolve(), html=True), name="react_build")
    uvicorn.run(app, host=host, port=port, log_level=log_level)