import json
import pathlib
import transformers
from threading import Thread
from fastapi.concurrency import asynccontextmanager
from sklearn.cluster import KMeans, DBSCAN
import uvicorn
from fastapi.staticfiles import StaticFiles
from typing import Literal, Callable, Any, Dict
import numpy as np
from fastapi import FastAPI, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tqdm import tqdm
import io
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn import manifold
from sklearn.decomposition import KernelPCA, PCA
from ..types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE, IMAGE_TYPE
from .. import metrics

# from ..redis_cache import redis_cache, redis_client

import tensorflow as tf
import keras as K
import tensorflow_datasets as tfds
from . import utils as utils
import umap
from ..channelexplorer import Server


class ChannelExplorer_TF(Server):
    def __init__(
        self,
        model: K.Model,
        inputs: Any,
        summary_fn_image: Callable[
            [IMAGE_BATCH_TYPE], SUMMARY_BATCH_TYPE
        ] = metrics.summary_fn_image_l2,
        summary_fn_dense: Callable[
            [DENSE_BATCH_TYPE], SUMMARY_BATCH_TYPE
        ] = metrics.summary_fn_dense_identity,
        log_level: Literal["info", "debug"] = "info",
        layers_to_show: list[str] | Literal["all"] = "all",
    ):
        """
        __init__ Creates a ChannelExplorer object for TensorFlow models.

        :param preprocess: The preprocessing function to preprocess the input image before feeding to the model. This will be run just before running the images through the model., defaults to lambdax:x
        :type preprocess: Callable, optional
        :param preprocess_inverse: This function is needed because the dataset is not directly stored in memory. To make it efficient, the preprocessed input is saved. So when displaying to the front-end, another function is needed to convert the input to image again for displaying. , defaults to lambdax:x
        :type preprocess_inverse: Callable, optional
        :return: The ChannelExplorer object.
        :rtype: ChannelExplorer
        """
        # redis_client.flushdb()

        self.model = model
        self.inputs = inputs
        self.log_level = log_level
        self.summary_fn_image = summary_fn_image
        self.summary_fn_dense = summary_fn_dense
        
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

        self.activations: dict[str, Any] = {}
        self.activationsSummary: dict[str, list[list[float]]] = {}

        # Add APIs
        @self.app.get("/api/model/")
        async def read_model():
            return self.model_graph

        # For async requests
        # @self.app.post("/api/analysis")
        # async def analysis(
        #     labels: list[int], examplePerClass: int = 5, shuffle: bool = False
        # ):
        #     task_id = task(cached_analysis, labels, examplePerClass, shuffle)
        #     return {"message": "Analysis started", "task_id": task_id}

        # def task(func, *args):
        #     task_id = utils.create_unique_task_id()
        #     thread = Thread(
        #         target=run_and_save_task_result, args=(task_id, func, *args, task_id)
        #     )
        #     thread.start()
        #     return task_id

        # def run_and_save_task_result(task_id: str, func, *args, **kwargs):
        #     result = func(*args, **kwargs)
        #     redis_client.setex(task_id, 3600, json.dumps(result))

        # @redis_cache(ttl=3600 * 8)
        # def cached_analysis(
        #     labels: list[int],
        #     examplePerClass: int = 5,
        #     shuffle: bool = False,
        #     _task_id: str = "",
        # ):
        #     return self._analysis(
        #         labels,
        #         examplePerClass,
        #         shuffle,
        #         progress=lambda x: redis_client.set(f"{_task_id}-progress", x),
        #     )

        # @self.app.get("/api/taskStatus")
        # async def taskStatus(task_id: str):
        #     result = redis_client.get(task_id)
        #     if result:
        #         return {
        #             "message": "Task completed",
        #             "task_id": task_id,
        #             "payload": json.loads(result),
        #         }
        #     result = redis_client.get(f"{task_id}-progress")
        #     if result:
        #         return {
        #             "message": result.decode("utf-8"),
        #             "task_id": task_id,
        #             "payload": None,
        #         }
        #     return {"message": "No Task found", "task_id": task_id, "payload": None}
        
        # Run the analysis
        self._analysis()

        @self.app.get("/api/analysis/layer/{layer}/argmax")
        async def get_argmax(layer: str):
            return np.argmax(self.activations[layer], axis=1).tolist()

        @self.app.get(
            "/api/analysis/image/{image_idx}/layer/{layer_name}/filter/{filter_index}"
        )
        async def get_activation_images(
            image_idx: int, layer_name: str, filter_index: int
        ):
            image = self.activations[layer_name][image_idx, :, :, filter_index]
            image -= image.min()
            image = (image - np.percentile(image, 10)) / (
                np.percentile(image, 90) - np.percentile(image, 10)
            )
            image = np.clip(image, 0, 1)
            image = 1 - image
            image = (image * 255).astype(np.uint8)
            image = np.stack((image,) * 3, axis=-1)
            img = Image.fromarray(image)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            return Response(content, headers=headers, media_type="image/png")

        @self.app.get("/api/analysis/layer/{layer_name}/embedding")
        async def analysisLayerEmbedding(
            layer_name: str,
            normalization: Literal["none", "row", "col"] = "none",
            method: Literal["mds", "tsne", "pca", "kpca", "umap", "autoencoder", 'autoencoder-pytorch'] = "umap",
            distance: Literal["euclidean", "jaccard"] = "euclidean",
            take_summary: bool = True,
        ):
            if take_summary:
                print("Using summary")
                this_activation = self.activationsSummary[layer_name]
            else:
                print("Not using summary")
                this_activation = self.activations[layer_name]
            this_activation = np.array(this_activation).reshape(len(this_activation), -1)

            if normalization == "row":
                this_activation = normalize(this_activation, axis=1, norm="l1")
            elif normalization == "col":
                this_activation = normalize(this_activation, axis=0, norm="l1")

            act_dist_mat = np.zeros((len(this_activation), len(this_activation)))

            for i, acti in tqdm(enumerate(this_activation), total=len(this_activation)):
                for j, actj in enumerate(this_activation):
                    if i == j:
                        act_dist_mat[i, j] = 0
                        continue
                    if i > j:
                        continue
                    if distance == "euclidean":
                        act_dist_mat[i, j] = utils.single_activation_distance(
                            acti, actj
                        )
                    elif distance == "jaccard":
                        act_dist_mat[i, j] = utils.single_activation_jaccard_distance(
                            acti, actj
                        )

                    act_dist_mat[j, i] = act_dist_mat[i, j]

            if method == 'pca':
                class EmbeddingModelPCA:
                    def __init__(self, input_dim):
                        self.embedding_model = PCA(n_components=2)
                    def fit_transform(self, x):
                        return self.embedding_model.fit_transform(this_activation)
                    
                embedding_model = EmbeddingModelPCA(this_activation.shape[1])

            elif method == 'kpca':
                embedding_model = KernelPCA(n_components=2, kernel='precomputed')
            elif method == "mds":
                embedding_model = manifold.MDS(
                    n_components=2,
                    dissimilarity="precomputed",
                    # random_state=6,
                    normalized_stress="auto",
                )
            elif method == "tsne":
                embedding_model = manifold.TSNE(
                    n_components=2,
                    metric="precomputed",
                    # random_state=6,
                    perplexity=min(30, len(this_activation) - 1),
                    init="random",
                )
            elif method == "umap":
                embedding_model = umap.UMAP(n_components=2, metric="precomputed")
            elif method == 'autoencoder':
                class EmbeddingModelAE:
                    def __init__(self, input_dim):
                        # Encoder
                        input_layer = K.Input(shape=(input_dim,))
                        encoded = K.layers.Dense(128, activation='relu')(input_layer)
                        encoded = K.layers.Dense(64, activation='relu')(encoded)
                        bottleneck = K.layers.Dense(2, activation='relu')(encoded)

                        # Decoder
                        decoded = K.layers.Dense(64, activation='relu')(bottleneck)
                        decoded = K.layers.Dense(128, activation='relu')(decoded)
                        decoded = K.layers.Dense(input_dim, activation='sigmoid')(decoded)

                        # Autoencoder Model
                        autoencoder = K.Model(input_layer, decoded)

                        # Encoder Model
                        encoder = K.Model(input_layer, bottleneck)

                        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
                        self.autoencoder = autoencoder
                        self.encoder = encoder

                    def fit_transform(self, x):
                        self.autoencoder.fit(this_activation, this_activation, epochs=50, batch_size=256, shuffle=True)
                        return self.encoder.predict(this_activation)
                
                embedding_model = EmbeddingModelAE(this_activation.shape[1])
            elif method == 'autoencoder-pytorch':
                import torch
                import torch.nn as nn
                import torch.optim as optim

                class EmbeddingModelAE(nn.Module):
                    def __init__(self, input_dim):
                        super(EmbeddingModelAE, self).__init__()
                        
                        # Encoder
                        self.encoder = nn.Sequential(
                            nn.Linear(input_dim, 128),
                            nn.ReLU(True),
                            nn.Linear(128, 64),
                            nn.ReLU(True),
                            nn.Linear(64, 2),
                            nn.ReLU(True)  # Bottleneck
                        )
                        
                        # Decoder
                        self.decoder = nn.Sequential(
                            nn.Linear(2, 64),
                            nn.ReLU(True),
                            nn.Linear(64, 128),
                            nn.ReLU(True),
                            nn.Linear(128, input_dim),
                            nn.Sigmoid()  # Reconstruction
                        )
                        
                    def forward(self, x):
                        x = self.encoder(x)
                        x = self.decoder(x)
                        return x

                    def fit_transform(self, x, num_epochs=5000, batch_size=256):
                        # get normalized this_activation
                        norm_this_activation = normalize(this_activation, axis=0, norm='l1')
                        # Assuming x is a NumPy array. Convert it to a PyTorch tensor.
                        x = torch.tensor(norm_this_activation, dtype=torch.float32)
                        
                        # DataLoader can be used here if you have a large dataset
                        optimizer = optim.Adam(self.parameters(), lr=1e-3)
                        criterion = nn.BCELoss()
                        
                        for epoch in range(num_epochs):
                            optimizer.zero_grad()
                            
                            # Forward pass
                            outputs = self(x)
                            loss = criterion(outputs, x)
                            
                            # Backward and optimize
                            loss.backward()
                            optimizer.step()
                            
                            if (epoch+1) % 10 == 0:
                                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                                
                        return self.encoder(x).detach().numpy()

                embedding_model = EmbeddingModelAE(this_activation.shape[1])


            coords = embedding_model.fit_transform(act_dist_mat)

            # makedir(f'output/analysis/layer/{layer_name}')
            # with open(f'output/analysis/layer/{layer_name}/embedding.json', 'w') as f:
            #     json.dump(coords.tolist(), f)

            return coords.tolist()

        @self.app.get("/api/analysis/alldistances")
        async def analysisAllDistances():
            act_dist_mat = np.zeros(
                (len(self.activationsSummary), len(self.activationsSummary))
            )
            for i, acti in tqdm(
                enumerate(self.activationsSummary), total=len(self.activationsSummary)
            ):
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
            this_activation = self.activationsSummary[layer_name]

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
            return self.activationsSummary[layer_name].tolist()

        @self.app.get("/api/analysis/layer/{layer_name}/{channel}/heatmap/{image_name}")
        async def analysisLayerHeatmapImage(
            layer_name: str, channel: int, image_name: int
        ):
            act_img = self.activations[layer_name][image_name, :, :, channel]
            act_img = (act_img - act_img.min()) / (act_img.max() - act_img.min())
            image = act_img
            image = (image * 255).astype(np.uint8)
            img = Image.fromarray(image)


            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            
            return Response(content, headers=headers, media_type="image/png")

        @self.app.get("/api/analysis/allembedding")
        async def analysisAllEmbedding():
            act_dist_mat = np.zeros(
                (len(self.activationsSummary), len(self.activationsSummary))
            )
            for i, acti in tqdm(
                enumerate(self.activationsSummary), total=len(self.activationsSummary)
            ):
                for j, actj in enumerate(self.activationsSummary):
                    if i == j:
                        act_dist_mat[i, j] = 0
                        continue
                    if i > j:
                        continue
                    act_dist_mat[i, j] = utils.activation_distance(acti, actj)
                    act_dist_mat[j, i] = act_dist_mat[i, j]

            mds = manifold.MDS(
                n_components=2, dissimilarity="precomputed", random_state=6
            )
            results = mds.fit(act_dist_mat)
            coords = results.embedding_

            # makedir(f'output/analysis')
            # with open(f'output/analysis/allembedding.json', 'w') as f:
            #     json.dump(coords.tolist(), f)

            return coords.tolist()

        @self.app.get("/api/analysis/layer/{layer_name}/{channel}/kernel")
        async def analysisLayerKernel(layer_name: str, channel: int):
            if len(model.get_layer(layer_name).get_weights()) == 0:
                # make a 3x3 identity kernel
                kernel = np.zeros((3, 3))
                kernel[1, 1] = 255
                kernel = kernel.astype(np.uint8)
            else:
                kernel = model.get_layer(layer_name).get_weights()[0][:, :, 0, channel]
                kernel = (
                    (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
                ).astype(np.uint8)
            img = Image.fromarray(kernel)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            # makedir(f'output/analysis/layer/{layer_name}/{channel}')
            # img.save(f'output/analysis/layer/{layer_name}/{channel}/kernel.png')
            return Response(content, headers=headers, media_type="image/png")

        @self.app.get("/api/total_inputs")
        async def total_inputs():
            return { 'total': len(self.inputs) }
        
        @self.app.get("/api/analysis/layer/{layer_name}/cluster")
        async def analysisLayerCluster(layer_name: str, outlier_threshold: float = 0.8, algorithm: Literal["kmeans", "dbscan"] = "kmeans", take_summary: bool = True):
            if take_summary:
                this_activation = self.activationsSummary[layer_name]
            else:
                this_activation = self.activations[layer_name]
                
            this_activation = np.array(this_activation).reshape(len(this_activation), -1)

            if algorithm == "kmeans":
                K = utils.find_optimal_k(this_activation)
                cluster_model = KMeans(n_clusters=K, n_init="auto")
                cluster_model.fit(this_activation)
            elif algorithm == "dbscan":
                cluster_model = DBSCAN(eps=0.3, min_samples=10)
                cluster_model.fit(this_activation)
                K = len(np.unique(cluster_model.labels_))

            # makedir(f'output/analysis/layer/{layer_name}')
            output = {
                "labels": cluster_model.labels_.tolist(),
                "outliers": [],
            }
            # with open(f'output/analysis/layer/{layer_name}/cluster.json', 'w') as f:
            #     json.dump(output, f)

            return output
        
    def _analysis(self):
        layers = list(
            map(
                lambda l: l.name,
                filter(
                    lambda l: isinstance(
                        l,
                        (
                            # K.layers.InputLayer,
                            K.layers.Conv2D,
                            K.layers.Dense,
                            K.layers.Flatten,
                            K.layers.Concatenate,
                            # transformer layers
                            K.layers.MultiHeadAttention,
                            K.layers.GlobalAveragePooling1D,
                            transformers.TFGPT2LMHeadModel,
                            transformers.TFGPT2MainLayer,
                            transformers.TFBertMainLayer,
                            transformers.TFBertModel,
                            transformers.TFBertForSequenceClassification,
                            transformers.TFBertForQuestionAnswering,
                            transformers.TFBertForTokenClassification,
                            transformers.TFBertForMultipleChoice,
                            transformers.TFBertForNextSentencePrediction,
                            transformers.TFBertForSequenceClassification,
                        ),
                    ),
                    self.model.layers,
                ),
            )
        )

        # Get activations
        def get_layer_activations(model, inputs, layers):
            print(inputs)
            # check if transformer (TFGPT2MainLayer) or CNN
            if isinstance(model.layers[0], transformers.TFGPT2MainLayer):
                # transformer
                outputs = model(inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                activations = [hidden_states[i] for i in layers]
            else:
                outputs = [model.get_layer(layer).output for layer in layers]
                activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
                activations = activation_model.predict(inputs)
            return activations

        def extract_activations(model, inputs, layers):
            activations = {}
            layer_outputs = get_layer_activations(model, inputs, layers)
            for layer_name, output in zip(layers, layer_outputs):
                activations[layer_name] = output
            return activations

        self.activations = extract_activations(self.model, self.inputs, layers)
        
        self.activationsSummary = {}
        for k, v in self.activations.items():
            if len(v[0].shape) == 1:
                # dense layer
                self.activationsSummary[k] = self.summary_fn_dense(v)
            elif len(v[0].shape) == 3:
                # Image layer
                self.summary_fn_image(v)
                self.activationsSummary[k] = self.summary_fn_image(v)