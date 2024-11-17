from typing import Any, Callable, Literal
from sklearn import manifold
from sklearn.decomposition import PCA, KernelPCA
from tqdm import tqdm
import umap
from PIL import Image
from io import BytesIO
from fastapi.responses import StreamingResponse

from sklearn.preprocessing import normalize

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from .Graph import GraphContainerNode, GraphFunctionNode, GraphTensorNode
from .metrics import summary_fn_dense_identity, summary_fn_image_l2, IMAGE_BATCH_TYPE, SUMMARY_BATCH_TYPE, DENSE_BATCH_TYPE

from .utils import get_model_graph, single_activation_distance, single_activation_jaccard_distance


class ChannelExplorer_Torch:
    def __init__(
        self,
        model: torch.nn.Module,
        all_inputs: list[Any],
        summary_fn_image: Callable[[IMAGE_BATCH_TYPE], SUMMARY_BATCH_TYPE] = summary_fn_image_l2,
        summary_fn_dense: Callable[[DENSE_BATCH_TYPE], SUMMARY_BATCH_TYPE] = summary_fn_dense_identity,
        log_level: Literal["info", "debug"] = "info",
        device: Literal["cpu", "cuda"] = "cuda",
    ):
        """
        __init__ Creates a ChannelExplorer object for PyTorch models.

        :param preprocess: The preprocessing function to preprocess the input image before feeding to the model. This will be run just before running the images through the model., defaults to lambdax:x
        :type preprocess: Callable, optional
        :param preprocess_inverse: This function is needed because the dataset is not directly stored in memory. To make it efficient, the preprocessed input is saved. So when displaying to the front-end, another function is needed to convert the input to image again for displaying. , defaults to lambdax:x
        :type preprocess_inverse: Callable, optional
        :return: The ChannelExplorer object.
        :rtype: ChannelExplorer
        """

        self.model = model
        self.all_inputs = all_inputs
        self.log_level = log_level
        self.summary_fn_image = summary_fn_image
        self.summary_fn_dense = summary_fn_dense
        self.device = device

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

        self.model_graph = get_model_graph(model, all_inputs, device, save_dot=True)
        
        # Add APIs
        @self.app.get("/api/model/")
        async def read_model():
            return self.get_model_graph()
        

        @self.app.get("/api/analysis/{tensor_id}/projection/")
        async def get_projection(
            tensor_id: str,
            normalization: Literal["none", "row", "col"] = "none",
            method: Literal["pca", "kpca", "mds", "tsne", "umap", "autoencoder", "autoencoder-pytorch"] = "pca",
            distance: Literal["euclidean", "jaccard"] = "euclidean"
        ):
            return self.get_projection(tensor_id, normalization=normalization, method=method, distance=distance)
        
        @self.app.get("/api/input/{index}")
        async def get_input(index: int):
            input_array = self.all_inputs[index]
            if torch.is_tensor(input_array):
                input_array = input_array.detach().cpu().numpy()
            
            input_array = np.transpose(input_array, (1,2,0))
            input_array = (input_array - input_array.min()) / (input_array.max() - input_array.min())
            input_array = (input_array * 255).astype(np.uint8)
            img = Image.fromarray(input_array)
            byte_io = BytesIO()
            img.save(byte_io, format='PNG')
            byte_io.seek(0)
            return StreamingResponse(byte_io, media_type="image/png")
        
        @self.app.get("/api/input/{index}/shape")
        async def get_input_shape(index: int):
            return self.all_inputs[index].shape
        
    def get_tensor_from_id(self, tensor_id: str):
        for node in self.model_graph.nodes:
            if isinstance(node, GraphTensorNode) and node.id == int(tensor_id):
                return node
        raise ValueError(f"Tensor with id {tensor_id} not found")
        
    def get_projection(
            self,
            tensor_id: str,
            take_summary: bool = True,
            normalization: Literal["none", "row", "col"] = "none",
            distance: Literal["euclidean", "jaccard"] = "euclidean",
            method: Literal["pca", "kpca", "mds", "tsne", "umap", "autoencoder", "autoencoder-pytorch"] = "pca"
        ):
        this_activation = self.get_tensor_from_id(tensor_id).value
        if len(this_activation.shape) != 4:
            raise ValueError(f"Expected activation shape [batch, channel, width, height], got {this_activation.shape}")
        if not isinstance(this_activation, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(this_activation)}")

        if take_summary:
            this_activation = self.summary_fn_image(this_activation)
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
                    act_dist_mat[i, j] = single_activation_distance(acti, actj)
                elif distance == "jaccard":
                    act_dist_mat[i, j] = single_activation_jaccard_distance(acti, actj)

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

        return coords.tolist()
    

    def get_model_graph(self):
        nodes = []
        for node in self.model_graph.nodes:
            if isinstance(node, GraphTensorNode):
                nodes.append({
                    "id": node.id,
                    "name": node.label,
                    "type": "tensor",
                    "shape": node.get_shape(),
                })
            elif isinstance(node, GraphContainerNode):
                nodes.append({
                    "id": node.id,
                    "name": node.label,
                    "type": "container",
                    "children": node.children,
                })
            elif isinstance(node, GraphFunctionNode):
                nodes.append({
                    "id": node.id,
                    "name": node.label,
                    "type": "function",
                    "input_shape": node.input_shape,
                    "output_shape": node.output_shape,
                })
            else:
                raise ValueError(f"Unknown node type: {type(node)}")
            
        edges = []
        for edge in self.model_graph.edges:
            edges.append({
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
            })

        return {
            "graph": {"nodes": nodes, "edges": edges}
        }

    def run_server(
            self,
            host: str = "localhost",
            port: int = 8000,
        ):
        # Starting the server
        # self.app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parents[0].joinpath('static').resolve(), html=True), name="react_build")
        uvicorn.run(self.app, host=host, port=port, log_level=self.log_level)

