import tensorflow as tf
from utils import get_model_layout, model_to_graph, pred_to_name, preprocess, remove_intermediate_node
import networkx as nx
import numpy as np
from PIL import Image
from io import BytesIO
import keract
import io

import uvicorn
from fastapi import FastAPI, File, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

currentModel = None
currentImg = None
currentActivations = None

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/api/models/")
async def read_model(model_name: str = 'inception_v3', pretrain_weights: str = 'imagenet'):
    global currentModel
    global currentImg
    if model_name == "inception_v3":
        currentModel = tf.keras.applications.inception_v3.InceptionV3(
            weights=pretrain_weights
        )

    currentModel.compile(loss="categorical_crossentropy", optimizer="adam")
    activation_pathway_full = model_to_graph(currentModel)
    simple_activation_pathway_full = remove_intermediate_node(activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['layer_type'] not in ['Conv2D', 'Dense', 'InputLayer', 'Concatenate'])

    node_pos = get_model_layout(simple_activation_pathway_full)

    # normalize node positions to be between 0 and 1
    min_x = min(pos[0] for pos in node_pos.values())
    max_x = max(pos[0] for pos in node_pos.values())
    min_y = min(pos[1] for pos in node_pos.values())
    max_y = max(pos[1] for pos in node_pos.values())

    node_pos = {
        node: ((pos[0] - min_x)/(max_x - min_x), (pos[1] - min_y)/(max_y - min_y)) for node, pos in node_pos.items()
    }

    for node, pos in node_pos.items():
        simple_activation_pathway_full.nodes[node]['pos'] = { 'x': pos[0], 'y': pos[1] }

    return nx.node_link_data(simple_activation_pathway_full)

@app.post("/api/exampleimg/")
async def example_img(file: bytes = File(...)):
    global currentModel
    global currentImg
    global currentActivations
    currentImg = np.array(Image.open(BytesIO(file)))
    processedImg = preprocess(currentImg, size=currentModel.input.shape[1:3].as_list())
    input_img: np.ndarray = tf.reshape(processedImg, [1, *processedImg.shape]).numpy()
    currentActivations = keract.get_activations(currentModel, input_img, layer_names=None, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)


    pred: tf.Tensor = currentModel(input_img)
    pred_label = pred_to_name(tf.math.argmax(pred[0]).numpy())
    return { "prediction": pred_label }

@app.get("/api/activations/{layer_name}")
async def get_activation(layer_name: str):
    global currentActivations
    images = currentActivations[layer_name]
    print(images.shape)
    return {
        "n_filters": images.shape[-1]
    }

@app.get("/api/activations/{layer_name}/image/{index}")
async def get_activations(layer_name: str, index: int):
    global currentActivations
    image = currentActivations[layer_name][0, :, :, index]

    image = np.abs(image)
    image = (image - np.percentile(image, 10)) / (np.percentile(image, 90) - np.percentile(image, 10))
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = np.stack((image,)*3, axis=-1)
    print(image.shape)
    img = Image.fromarray(image)
    print(img)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        content = output.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(content, headers=headers, media_type='image/png')

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000, log_level="info")