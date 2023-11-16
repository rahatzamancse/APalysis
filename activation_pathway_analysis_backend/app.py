from functools import lru_cache
from typing import Literal
from pydantic import BaseModel
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

import activation_pathway_analysis_backend.utils_tf as utils_tf
import networkx as nx
import numpy as np
from PIL import Image
import keract
import io
from tqdm import tqdm
from sklearn import manifold

import uvicorn
from fastapi import FastAPI, File, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import normalize


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
    activation_pathway_full = utils_tf.tensorflow_model_to_graph(model)
    simple_activation_pathway_full = utils_tf.remove_intermediate_node(
        activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['layer_type'] not in ['Conv2D', 'Dense', 'InputLayer', 'Concatenate'])

    node_pos = utils_tf.get_model_layout(simple_activation_pathway_full)

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
        simple_activation_pathway_full.nodes[node]['pos'] = {
            'x': pos[0], 'y': pos[1]}
    
    # Edge weights
    kernel_norms = {}
    for layer_id, layer_data in simple_activation_pathway_full.nodes(data=True):
        if layer_data['layer_type'] in ['Concatenate', 'InputLayer', 'Dense']:
            continue
        layer_name = layer_data['name']
        kernel = model.get_layer(layer_name).get_weights()
        kernel_norm = np.linalg.norm(kernel[0], axis=(0,1), ord=2).sum(axis=0)
        kernel_norms[layer_name] = kernel_norm.tolist()
        # kernel = ((kernel - kernel.min()) / (kernel.max() -
        #           kernel.min()) * 255).astype(np.uint8)
        # img = Image.fromarray(kernel)
        
    output = {
        'graph': nx.node_link_data(simple_activation_pathway_full),
        'meta': {
            'depth': max(nx.shortest_path_length(simple_activation_pathway_full, next(n for n, d in simple_activation_pathway_full.nodes(data=True) if d['layer_type'] == 'InputLayer')).values())
        },
        'edge_weights': kernel_norms,
    }
    
    # with open(f'output/model.json', 'w') as f:
    #     json.dump(output, f)

    return output


@app.get("/api/labels/")
async def read_labels():
    global labels
    global dataset_info

    output = None
    if labels:
        output = labels.names
    else:
        output = list(dataset_info.features['label'].names)
        
    # with open(f'output/labels.json', 'w') as f:
    #     json.dump(output, f)

    return output

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
    feature_hunt_image = utils_tf.preprocess(
        (img, -1), size=model.input.shape[1:3].as_list())[0].numpy()

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
    )), model.layers)))
    activation = keract.get_activations(model, np.array(
        [feature_hunt_image]), layer_names=layers, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)

    mask_img = utils_tf.get_mask_img(
        [[p.x, p.y] for p in points], feature_hunt_image.shape[0:2])

    feature_hunt_activated_channels = utils_tf.get_mask_activation_channels(
        mask_img, activation, summary_fn_image)

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

def makedir(path):
    # make a directory if it doesn't exist
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        
# import cv2

@app.post("/api/analysis")
async def analysis(labels: list[int], examplePerClass: int = 50, shuffle: bool = False):
    return run_model_on_input(tuple(labels), examplePerClass, shuffle)
    
@lru_cache
def run_model_on_input(labels, examplePerClass: int = 50, shuffle: bool = False):
    global datasetImgs
    global activations
    global activationsSummary
    global datasetLabels
    global selectedLabels
    global shuffled
    global predictions

    shuffled = shuffle
    labels = list(labels)
    selectedLabels = labels

    layers = list(map(lambda l: l.name, filter(lambda l: isinstance(l, (
        # tf.keras.layers.InputLayer,
        tf.keras.layers.Conv2D,
        tf.keras.layers.Dense,
        tf.keras.layers.Flatten,
        tf.keras.layers.Concatenate,
    )), model.layers)))
    

    datasetImgs = [[] for _ in range(len(labels))]
    activations = [[] for _ in range(len(labels))]
    activationsSummary = [[] for _ in range(len(labels))]
    datasetLabels = [[] for _ in range(len(labels))]

    @tf.function
    def filter_by_labels(img, label):
        return tf.reduce_any(tf.equal(label, labels))
        
    for i, (img, label) in tqdm(enumerate(
    utils_tf.shuffle_or_noshuffle(dataset, shuffle=shuffled
    ).filter(filter_by_labels
    ).map(preprocess
    ).batch(1
    )), total=examplePerClass*len(labels)):

        if len(datasetImgs[labels.index(label)]) >= examplePerClass:
            continue

        label_idx = labels.index(label)

        # Get activations
        activation = keract.get_activations(
            model, img, layer_names=layers, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)

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
        
        # path = f'../../saved_data/{MODEL}_{DATASET}/class-{labels[label_idx]}/{i}'
        # makedir(path)

        # bgr_image = cv2.cvtColor(utils.rescale_img(img.numpy()), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(path + f'/orig.png', bgr_image)
        # for layer, acts in activation.items():
        #     for batch, act in enumerate(acts):
        #         try:
        #             for i, x in enumerate(act.transpose(2,0,1)):
        #                 makedir(f'{path}/{layer}')
        #                 # cv2.imwrite(f'{path}/{layer}/{i}.png', utils.rescale_img(x))
        #         except Exception as e:
        #             print('ERROR: act', act.shape)
        #             print(e)


    datasetImgs = [j for i in datasetImgs for j in i]
    activations = [j for i in activations for j in i]
    activationsSummary = [j for i in activationsSummary for j in i]
    datasetLabels = [j for i in datasetLabels for j in i]

    # Get the prediction with argmax
    predictions = []
    for i in range(len(activations)):
        predictions.append(np.argmax(activations[i][layers[-1]][0]).item())
        
    output = {
        "selectedClasses": selectedLabels,
        "examplePerClass": len(datasetImgs) // len(selectedLabels),
        "shuffled": shuffled,
        "predictions": predictions,
    }
    
    # with open(f'output/analysis.json', 'w') as f:
    #     json.dump(output, f)

    return output

@app.get("/api/analysis/layer/{layer}/argmax")
async def get_argmax(layer: str):
    global activations
    output = [np.argmax(activation[layer][0]).item() for activation in activations]
    
    # makedir(f'output/analysis/layer/{layer}')
    # with open(f'output/analysis/layer/{layer}/argmax.json', 'w') as f:
    #     json.dump(output, f)
    return output


@app.get("/api/analysis/image/{image_idx}/layer/{layer_name}/filter/{filter_index}")
async def get_activation_images(image_idx: int, layer_name: str, filter_index: int):
    image = activations[image_idx][layer_name][0, :, :, filter_index]
    image -= image.min()
    image = (image - np.percentile(image, 10)) / \
    (np.percentile(image, 90) - np.percentile(image, 10))
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
    
    # write img
    # dir = f'output/activation_images/{image_idx}/{layer_name}'
    # path = dir + f"/{filter_index}.png"
    # makedir(dir)
    # img.save(path)
    
    return Response(content, headers=headers, media_type='image/png')


@app.get("/api/analysis/layer/{layer_name}/embedding")
async def analysisLayerEmbedding(layer_name: str, normalization: Literal['none', 'row', 'col'] = 'none', method: Literal['mds', 'tsne'] = "mds", distance: Literal['euclidean', 'jaccard'] = "euclidean"):
    global activationsSummary
    this_activation = [activation[layer_name]
                       for activation in activationsSummary]
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
                act_dist_mat[i, j] = utils_tf.single_activation_distance(acti, actj)
            elif distance == 'jaccard':
                act_dist_mat[i, j] = utils_tf.single_activation_jaccard_distance(
                    acti, actj)

            act_dist_mat[j, i] = act_dist_mat[i, j]

    if method == 'mds':
        embedding_model = manifold.MDS(
            n_components=2, dissimilarity="precomputed", random_state=6, normalized_stress='auto')
    elif method == 'tsne':
        embedding_model = manifold.TSNE(n_components=2, metric='precomputed', random_state=6, perplexity=min(
            30, len(this_activation)-1), init='random')
    coords = embedding_model.fit_transform(act_dist_mat)
    
    # makedir(f'output/analysis/layer/{layer_name}')
    # with open(f'output/analysis/layer/{layer_name}/embedding.json', 'w') as f:
    #     json.dump(coords.tolist(), f)

    return coords.tolist()


@app.get("/api/analysis/alldistances")
async def analysisAllDistances():
    global activationsSummary

    act_dist_mat = np.zeros((len(activationsSummary), len(activationsSummary)))
    for i, acti in tqdm(enumerate(activationsSummary), total=len(activationsSummary)):
        for j, actj in enumerate(activationsSummary):
            if i == j:
                act_dist_mat[i, j] = 0
                continue
            if i > j:
                continue
            act_dist_mat[i, j] = utils_tf.activation_distance(acti, actj)
            act_dist_mat[j, i] = act_dist_mat[i, j]
            
    # makedir(f'output/analysis')
    # with open(f'output/analysis/alldistances.json', 'w') as f:
    #     json.dump(act_dist_mat.tolist(), f)

    return act_dist_mat.tolist()


@app.get("/api/analysis/layer/{layer_name}/embedding/distance")
async def analysisLayerEmbeddingDistance(layer_name: str):
    global activationsSummary
    this_activation = [activation[layer_name]
                       for activation in activationsSummary]
    this_activation = np.array(this_activation)

    act_dist_mat = np.zeros((len(this_activation), len(this_activation)))

    for i, acti in tqdm(enumerate(this_activation), total=len(this_activation)):
        for j, actj in enumerate(this_activation):
            if i == j:
                act_dist_mat[i, j] = 0
                continue
            if i > j:
                continue
            act_dist_mat[i, j] = utils_tf.single_activation_distance(acti, actj)
            act_dist_mat[j, i] = act_dist_mat[i, j]
            
    # makedir(f'output/analysis/layer/{layer_name}/embedding')
    # with open(f'output/analysis/layer/{layer_name}/embedding/distance.json', 'w') as f:
    #     json.dump(act_dist_mat.tolist(), f)

    return act_dist_mat.tolist()


@app.get("/api/analysis/layer/{layer_name}/heatmap")
async def analysisLayerHeatmap(layer_name: str):
    global activationsSummary
    # classActivations = {}
    # labels = np.load('labels_imagenette.npz')['labels']
    # activationSummary = np.load('activationSummary_imagenette.npz')
    layer_activation = [activation[layer_name].tolist()
                        for activation in activationsSummary]
    
    # makedir(f'output/analysis/layer/{layer_name}')
    # with open(f'output/analysis/layer/{layer_name}/heatmap.json', 'w') as f:
    #     json.dump(layer_activation, f)
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
            act_dist_mat[i, j] = utils_tf.activation_distance(acti, actj)
            act_dist_mat[j, i] = act_dist_mat[i, j]

    mds = manifold.MDS(
        n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(act_dist_mat)
    coords = results.embedding_
    
    # makedir(f'output/analysis')
    # with open(f'output/analysis/allembedding.json', 'w') as f:
    #     json.dump(coords.tolist(), f)

    return coords.tolist()


@app.get("/api/analysis/images/{index}")
async def inputImages(index: int):
    global datasetImgs
    global datasetLabels
    if index < 0 or index >= len(datasetImgs):
        return Response(status_code=404)
    image, _ = preprocess_inv(datasetImgs[index][0], datasetLabels[index])
    img = Image.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    
    # makedir(f'output/analysis/images/{index}')
    # img.save(f'output/analysis/images/{index}/orig.png')
    return Response(content, headers=headers, media_type='image/png')


@app.get("/api/analysis/layer/{layer_name}/{channel}/heatmap/{image_name}")
async def analysisLayerHeatmap(layer_name: str, channel: int, image_name: int):
    global activations
    global datasetImgs

    image = utils_tf.get_activation_overlay(datasetImgs[image_name][0].squeeze(
    ), activations[image_name][layer_name][0][:, :, channel], alpha=0.8)

    image = ((image / 2 + 0.5) * 255).astype(np.uint8)
    # image = (datasetImgs[index][0]*255).astype(np.uint8)
    img = Image.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    
    # makedir(f'output/analysis/layer/{layer_name}/{channel}/heatmap/{image_name}')
    # img.save(f'output/analysis/layer/{layer_name}/{channel}/heatmap/{image_name}/heatmap.png')
    return Response(content, headers=headers, media_type='image/png')

@app.get("/api/analysis/layer/{layer_name}/{channel}/kernel")
async def analysisLayerKernel(layer_name: str, channel: int):
    kernel = model.get_layer(layer_name).get_weights()[0][:, :, 0, channel]
    kernel = ((kernel - kernel.min()) / (kernel.max() -
              kernel.min()) * 255).astype(np.uint8)
    img = Image.fromarray(kernel)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    # makedir(f'output/analysis/layer/{layer_name}/{channel}')
    # img.save(f'output/analysis/layer/{layer_name}/{channel}/kernel.png')
    return Response(content, headers=headers, media_type='image/png')

@app.get("/api/analysis/layer/{layer_name}/cluster")
async def analysisLayerCluster(layer_name: str, outlier_threshold: float = 0.8):
    global activationsSummary
    global selectedLabels

    this_activation = [activation[layer_name]
                       for activation in activationsSummary]
    this_activation = np.array(this_activation)

    kmeans = KMeans(n_clusters=len(selectedLabels), n_init="auto")
    kmeans.fit(this_activation)
    
    distance_from_center = kmeans.transform(this_activation).min(axis=1)

    # average distance from center for each label
    mean_distance_from_center = np.zeros(len(selectedLabels))
    max_distance_from_center = np.zeros(len(selectedLabels))
    std_distance_form_center = np.zeros(len(selectedLabels))
    for i, label in enumerate(selectedLabels):
        mean_distance_from_center[i] = distance_from_center[kmeans.labels_ == i].mean()
        max_distance_from_center[i] = distance_from_center[kmeans.labels_ == i].max()
        std_distance_form_center[i] = distance_from_center[kmeans.labels_ == i].std()
        
    # https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf
    outliers = []
    for i in range(len(distance_from_center)):
        if distance_from_center[i] > mean_distance_from_center[kmeans.labels_[i]] + std_distance_form_center[kmeans.labels_[i]] * outlier_threshold:
            outliers.append(i)
            
    # makedir(f'output/analysis/layer/{layer_name}')
    output = {
        'labels': kmeans.labels_.tolist(),
        'centers': kmeans.cluster_centers_.tolist(),
        'distances': distance_from_center.tolist(),
        'outliers': outliers,
    }
    # with open(f'output/analysis/layer/{layer_name}/cluster.json', 'w') as f:
    #     json.dump(output, f)

    return output
    



@app.get("/api/analysis/predictions")
async def analysisPredictions():
    global predictions
    # makedir(f'output/analysis')
    # with open(f'output/analysis/predictions.json', 'w') as f:
    #     json.dump(predictions, f)
    return predictions


@app.get("/api/loaded_analysis")
async def loadedAnalysis():
    global datasetImgs
    global datasetLabels
    global selectedLabels
    global shuffled
    global predictions

    output = None
    if not selectedLabels:
        output = {
            "selectedClasses": [],
            "examplePerClass": 0,
        }
    else:
        output = {
            "selectedClasses": selectedLabels,
            "examplePerClass": len(datasetImgs) // len(selectedLabels),
            "shuffled": shuffled,
            "predictions": predictions,
        }
    
    # with open(f'output/loaded_analysis.json', 'w') as f:
    #     json.dump(output, f)
        
    return output


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
MODEL, DATASET = 'vgg16', 'imagenette'

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
        x = utils_tf.preprocess(x, y, size=inception_input_shape)

    def preprocess_inv(x, y):
        x = ((x / 2 + 0.5) * 255).astype(np.uint8).squeeze()
        return x, y

if __name__ == '__main__':
    host = "localhost"
    port = 8000
    log_level = "info"

    # Starting the server
    # app.mount("/", StaticFiles(directory='static', html=True), name="static")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
else:
    app.mount("/", StaticFiles(directory='static', html=True), name="static")