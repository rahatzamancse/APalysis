import json
import pathlib
from threading import Thread
from fastapi.concurrency import asynccontextmanager
from sklearn.cluster import KMeans
import uvicorn
from fastapi.staticfiles import StaticFiles
from typing import Literal, Callable, Any, Tuple
import numpy as np
from fastapi import FastAPI, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tqdm import tqdm
import io
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn import manifold
from .types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE, IMAGE_TYPE
from . import metrics

from .redis_cache import redis_cache, redis_client

import tensorflow as tf
from tensorflow import keras as K
import tensorflow_datasets as tfds
import keract
from . import utils_tf as utils


class APAnalysisTensorflowModel:
    def __init__(
        self,
        model: K.Model,
        dataset: tf.data.Dataset,
        label_names: list[str] = [],
        preprocess: Callable = lambda x: x,
        # TODO: remove preprocess_inverse and keep the original input data
        preprocess_inverse: Callable = lambda x: x,
        summary_fn_image: Callable[[IMAGE_BATCH_TYPE], SUMMARY_BATCH_TYPE] = metrics.summary_fn_image_l2,
        summary_fn_dense: Callable[[DENSE_BATCH_TYPE], SUMMARY_BATCH_TYPE] = metrics.summary_fn_dense_identity,
        log_level: Literal["info", "debug"] = "info",
        layers_to_show: list[str] | Literal["all"] = "all",
    ):
        redis_client.flushdb()

        self.model = model
        self.dataset = dataset
        self.log_level = log_level
        self.summary_fn_image = summary_fn_image
        self.summary_fn_dense = summary_fn_dense
        self.preprocess = preprocess
        self.preprocess_inv = preprocess_inverse

        # Setup caching
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield

        self.app = FastAPI(lifespan=lifespan)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.model_graph = utils.parse_model_graph(model, layers_to_show)
        self.shuffled = False
        self.label_names = label_names
        self.labels: list[int] = []
        self.selectedLabels: list[int] = []
        self.datasetImgs: list[list[IMAGE_TYPE]] = []
        self.activations: list[dict] = []
        self.activationsSummary: list[dict[str, float]] = []
        self.datasetLabels: list[list[int]] = []
        self.predictions: list[int] = []
        self.layer_activation_bounds: dict[str, Tuple[float, float]] = {}
        
        # Add APIs
        @self.app.get("/api/model/")
        async def read_model():
            return self.model_graph

        @self.app.get("/api/labels/")
        async def read_labels():
            return self.label_names

        @self.app.post("/api/analysis")
        async def analysis(labels: list[int], examplePerClass: int = 5, shuffle: bool = False):
            task_id = utils.create_unique_task_id()
            task(task_id, cached_analysis, labels, examplePerClass, shuffle, task_id)
            return {"message": "Analysis started", "task_id": task_id}

        def task(task_id: str, func, *args):
            thread = Thread(target=run_and_save_task_result, args=(task_id, func, *args))
            thread.start()
                
        def run_and_save_task_result(task_id: str, func, *args, **kwargs):
            result = func(*args, **kwargs)
            redis_client.setex(task_id, 3600, json.dumps(result))

        @redis_cache(ttl=3600*8)
        def cached_analysis(labels: list[int], examplePerClass: int = 5, shuffle: bool = False, _task_id: str = ""):
            return self._analysis(labels, examplePerClass, shuffle, progress=lambda x: redis_client.set(f"{_task_id}-progress", x))

        @self.app.get("/api/taskStatus")
        async def taskStatus(task_id: str):
            result = redis_client.get(task_id)
            if result:
                return {
                    "message": "Task completed",
                    "task_id": task_id,
                    "payload": json.loads(result)
                }
            result = redis_client.get(f"{task_id}-progress")
            if result:
                return {
                    "message": result.decode('utf-8'),
                    "task_id": task_id,
                    "payload": None
                }
            return {"message": "No Task found", "task_id": task_id, "payload": None }

        @self.app.get("/api/analysis/layer/{layer}/argmax")
        async def get_argmax(layer: str):
            return [np.argmax(activation[layer][0]).item() for activation in self.activations]

        @self.app.get("/api/analysis/image/{image_idx}/layer/{layer_name}/filter/{filter_index}")
        async def get_activation_images(image_idx: int, layer_name: str, filter_index: int):
            image = self.activations[image_idx][layer_name][0, :, :, filter_index]

            # save this numpy image
            # utils.makedir(f'output/activation-heatmap/image_{image_idx}/layer_{layer_name}')
            # np.save(f'output/activation-heatmap/image_{image_idx}/layer_{layer_name}/filter_{filter_index}.npy', image)

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
            return Response(content, headers=headers, media_type='image/png')

        @self.app.get("/api/analysis/layer/{layer_name}/embedding")
        async def analysisLayerEmbedding(layer_name: str, normalization: Literal['none', 'row', 'col'] = 'none', method: Literal['mds', 'tsne'] = "mds", distance: Literal['euclidean', 'jaccard'] = "euclidean"):
            this_activation = [activation[layer_name]
                               for activation in self.activationsSummary]
            this_activation = np.array(this_activation)

            if normalization == 'row':
                this_activation = normalize(this_activation, axis=1, norm='l1')
            elif normalization == 'col':
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
                        act_dist_mat[i, j] = utils.single_activation_distance(acti, actj)
                    elif distance == 'jaccard':
                        act_dist_mat[i, j] = utils.single_activation_jaccard_distance(
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

        @self.app.get("/api/analysis/alldistances")
        async def analysisAllDistances():
            act_dist_mat = np.zeros((len(self.activationsSummary), len(self.activationsSummary)))
            for i, acti in tqdm(enumerate(self.activationsSummary), total=len(self.activationsSummary)):
                for j, actj in enumerate(self.activationsSummary):
                    if i == j:
                        act_dist_mat[i, j] = 0
                        continue
                    if i > j:
                        continue
                    act_dist_mat[i, j] = utils.activation_distance(acti, actj)
                    act_dist_mat[j, i] = act_dist_mat[i, j]
                    
            # makedir(f'output/analysis')
            # with open(f'output/analysis/alldistances.json', 'w') as f:
            #     json.dump(act_dist_mat.tolist(), f)

            return act_dist_mat.tolist()


        @self.app.get("/api/analysis/layer/{layer_name}/embedding/distance")
        async def analysisLayerEmbeddingDistance(layer_name: str):
            this_activation = [activation[layer_name]
                               for activation in self.activationsSummary]
            this_activation = np.array(this_activation)

            act_dist_mat = np.zeros((len(this_activation), len(this_activation)))

            for i, acti in tqdm(enumerate(this_activation), total=len(this_activation)):
                for j, actj in enumerate(this_activation):
                    if i == j:
                        act_dist_mat[i, j] = 0
                        continue
                    if i > j:
                        continue
                    act_dist_mat[i, j] = utils.single_activation_distance(acti, actj)
                    act_dist_mat[j, i] = act_dist_mat[i, j]
                    
            # makedir(f'output/analysis/layer/{layer_name}/embedding')
            # with open(f'output/analysis/layer/{layer_name}/embedding/distance.json', 'w') as f:
            #     json.dump(act_dist_mat.tolist(), f)

            return act_dist_mat.tolist()


        @self.app.get("/api/analysis/layer/{layer_name}/heatmap")
        async def analysisLayerHeatmap(layer_name: str):
            return [activation[layer_name].tolist() for activation in self.activationsSummary]

        @self.app.get("/api/analysis/layer/{layer_name}/{channel}/heatmap/{image_name}")
        async def analysisLayerHeatmapImage(layer_name: str, channel: int, image_name: int):
            in_img = self.datasetImgs[image_name][0].squeeze()
            in_img = (in_img - in_img.min()) / (in_img.max() - in_img.min())

            act_img = self.activations[image_name][layer_name][0][:, :, channel].squeeze()
            act_img = (act_img - self.layer_activation_bounds[layer_name][0]) / (self.layer_activation_bounds[layer_name][1] - self.layer_activation_bounds[layer_name][0])
            print(act_img.min(), act_img.max())
            image = utils.get_activation_overlay(
                in_img,
                act_img,
                alpha=0.6
            )
            image = (image * 255).astype(np.uint8)
            img = Image.fromarray(image)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
                
            # save the image
            utils.makedir(f'output/activations/{layer_name}/{channel}')
            img.save(f'output/activations/{layer_name}/{channel}/act_{image_name}.png')

            headers = {'Content-Disposition': 'inline; filename="test.png"'}
            return Response(content, headers=headers, media_type='image/png')


        @self.app.get("/api/analysis/allembedding")
        async def analysisAllEmbedding():
            act_dist_mat = np.zeros((len(self.activationsSummary), len(self.activationsSummary)))
            for i, acti in tqdm(enumerate(self.activationsSummary), total=len(self.activationsSummary)):
                for j, actj in enumerate(self.activationsSummary):
                    if i == j:
                        act_dist_mat[i, j] = 0
                        continue
                    if i > j:
                        continue
                    act_dist_mat[i, j] = utils.activation_distance(acti, actj)
                    act_dist_mat[j, i] = act_dist_mat[i, j]

            mds = manifold.MDS(
                n_components=2, dissimilarity="precomputed", random_state=6)
            results = mds.fit(act_dist_mat)
            coords = results.embedding_
            
            # makedir(f'output/analysis')
            # with open(f'output/analysis/allembedding.json', 'w') as f:
            #     json.dump(coords.tolist(), f)

            return coords.tolist()


        @self.app.get("/api/analysis/images/{index}")
        async def inputImages(index: int):
            if index < 0 or index >= len(self.datasetImgs):
                return Response(status_code=404)
            image, _ = self.preprocess_inv(self.datasetImgs[index][0], self.datasetLabels[index])
            img = Image.fromarray(image)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {'Content-Disposition': 'inline; filename="test.png"'}
            
            # makedir(f'output/analysis/images/{index}')
            # img.save(f'output/analysis/images/{index}/orig.png')
            return Response(content, headers=headers, media_type='image/png')


        @self.app.get("/api/analysis/layer/{layer_name}/{channel}/kernel")
        async def analysisLayerKernel(layer_name: str, channel: int):
            if len(model.get_layer(layer_name).get_weights()) == 0:
                # make a 3x3 identity kernel
                kernel = np.zeros((3, 3))
                kernel[1, 1] = 255
                kernel = kernel.astype(np.uint8)
            else:
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

        @self.app.get("/api/analysis/layer/{layer_name}/cluster")
        async def analysisLayerCluster(layer_name: str, outlier_threshold: float = 0.8):
            this_activation = [activation[layer_name]
                               for activation in self.activationsSummary]
            this_activation = np.array(this_activation)

            kmeans = KMeans(n_clusters=len(self.selectedLabels), n_init="auto")
            kmeans.fit(this_activation)
            
            distance_from_center = kmeans.transform(this_activation).min(axis=1)

            # average distance from center for each label
            mean_distance_from_center = np.zeros(len(self.selectedLabels))
            max_distance_from_center = np.zeros(len(self.selectedLabels))
            std_distance_form_center = np.zeros(len(self.selectedLabels))
            for i, label in enumerate(self.selectedLabels):
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

        @self.app.get("/api/analysis/predictions")
        async def analysisPredictions():
            return self.predictions


        @self.app.get("/api/loaded_analysis")
        async def loadedAnalysis():
            output = None
            if not self.selectedLabels:
                output = {
                    "selectedClasses": [],
                    "examplePerClass": 0,
                }
            else:
                output = {
                    "selectedClasses": self.selectedLabels,
                    "examplePerClass": len(self.datasetImgs) // len(self.selectedLabels),
                    "shuffled": self.shuffled,
                    "predictions": self.predictions,
                }
            return output
        
        # Save all input images
        @self.app.post("/api/analysis/images/save")
        async def saveInputImages():
            # delete existing input_images
            utils.delete_dir('output/input_images')
            # create a directory for each class, then save the images
            examplePerClass = len(self.datasetImgs) // len(self.selectedLabels)
            for i, label in enumerate(self.selectedLabels):
                utils.makedir(f'output/input_images/{label}')
                for j, img in enumerate(self.datasetImgs[examplePerClass*i:examplePerClass*(i+1)]):
                    img, _ = self.preprocess_inv(img, label)
                    img = Image.fromarray(img.squeeze())
                    img.save(f'output/input_images/{label}/{j}.png')
            
            # zip the directory and 
            utils.zip_dir('output/input_images', 'dataset')
            
            # return the zip file
            response = FileResponse(path='dataset.zip', filename='dataset.zip', media_type='application/zip')
            return response
                
    def _analysis(self, labels: list[int], examplePerClass: int = 50, shuffle: bool = False, progress: Callable[[str], Any] = lambda x: None):
        progress("Starting the analysis")
        self.shuffled = shuffle
        self.labels = list(labels)
        self.selectedLabels = labels

        def get_layer_name(layer: K.layers.Layer):
            return layer.name
        layers : list[str] = list(map(get_layer_name, filter(lambda l: isinstance(l, (
            # K.layers.InputLayer,
            K.layers.Conv2D,
            K.layers.Dense,
            K.layers.Flatten,
            K.layers.Concatenate,
        )), self.model.layers)))
        
        __datasetImgs = [[] for _ in range(len(labels))]
        __activations = [[] for _ in range(len(labels))]
        __activationsSummary = [[] for _ in range(len(labels))]
        __datasetLabels = [[] for _ in range(len(labels))]

        @tf.function
        def filter_by_labels(img, label):
            return tf.reduce_any(tf.equal(label, labels))
            
        for i, (img, label) in tqdm(enumerate(
        utils.shuffle_or_noshuffle(self.dataset, shuffle=self.shuffled
        ).filter(filter_by_labels
        ).map(self.preprocess
        ).batch(1
        )), total=examplePerClass*len(labels)):
            if len(__datasetImgs[labels.index(label)]) >= examplePerClass:
                continue

            progress(f"Processing image {i}/{examplePerClass*len(labels)}")

            label_idx = labels.index(label)

            # Get activations
            activation = keract.get_activations(
                self.model, img, layer_names=layers, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
            
            __datasetImgs[label_idx].append(img.numpy())
            __activations[label_idx].append(activation)

            activationSummary = {}
            for k, v in activation.items():
                if len(v[0].shape) == 1:
                    # dense layer
                    activationSummary[k] = self.summary_fn_dense(v)[0]
                elif len(v[0].shape) == 3:
                    # Image layer
                    self.summary_fn_image(v)
                    activationSummary[k] = self.summary_fn_image(v)[0]
            __activationsSummary[label_idx].append(activationSummary)

            __datasetLabels[label_idx].append(label.numpy()[0].item())

            if all((len(dtImgs) >= examplePerClass) for dtImgs in __datasetImgs):
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

        self.datasetImgs = [j for i in __datasetImgs for j in i]
        self.activations = [j for i in __activations for j in i]
        self.activationsSummary = [j for i in __activationsSummary for j in i]
        self.datasetLabels = [j for i in __datasetLabels for j in i]
        
        # calculate self.layer_activation_bounds for each layer
        for layer in layers:
            layer_activations = [activation[layer] for activation in self.activations]
            self.layer_activation_bounds[layer] = (
                np.percentile(layer_activations, 2),
                np.percentile(layer_activations, 98),
            )
            
        # Get the prediction with argmax
        self.predictions = []
        for i in range(len(self.activations)):
            self.predictions.append(np.argmax(self.activations[i][layers[-1]][0]).item())
            
        output = {
            "selectedClasses": self.selectedLabels,
            "examplePerClass": len(self.datasetImgs) // len(self.selectedLabels),
            "shuffled": self.shuffled,
            "predictions": self.predictions,
        }
        
        # with open(f'output/analysis.json', 'w') as f:
        #     json.dump(output, f)

        return output



    def run_server(
            self,
            host: str = "0.0.0.0",
            port: int = 8000,
        ):
        # Starting the server
        # self.app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parents[0].joinpath('static').resolve(), html=True), name="react_build")
        uvicorn.run(self.app, host=host, port=port, log_level=self.log_level)
