import json
import pathlib
import transformers
from threading import Thread
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
from sklearn.cluster import KMeans, DBSCAN
import uvicorn
from fastapi.staticfiles import StaticFiles
from typing import Literal, Callable, Any, Dict, Tuple
import numpy as np
from fastapi import FastAPI, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tqdm import tqdm
import io
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn import manifold
from ..types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE, IMAGE_TYPE
from .. import metrics
import networkx as nx

import torch
from . import utils as utils
import umap
from ..channelexplorer import Server

class ChannelExplorer_Torch(Server):
    def __init__(
        self,
        models: list[torch.nn.Module],
        all_inputs: list[Any],
        summary_fn_image: Callable[[IMAGE_BATCH_TYPE], SUMMARY_BATCH_TYPE] = metrics.summary_fn_image_l2,
        summary_fn_dense: Callable[[DENSE_BATCH_TYPE], SUMMARY_BATCH_TYPE] = metrics.summary_fn_dense_identity,
        log_level: Literal["info", "debug"] = "info",
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

        self.models = models
        self.all_inputs = all_inputs
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

        self.model_graph = utils.extract_activations_graphs(models, all_inputs)
        
        # add expanded = false to all nodes
        hierarchy_graph = self.model_graph.copy()
        edges_to_remove = [(u, v) for u, v, d in hierarchy_graph.edges(data=True) if d.get("edge_type") == "data_flow"]
        hierarchy_graph.remove_edges_from(edges_to_remove)
        for node in self.model_graph.nodes(data=True):
            node[1]['expanded'] = False
            is_leaf = hierarchy_graph.in_degree(node[0]) == 0
            node[1]['is_leaf'] = is_leaf

        # Add APIs
        @self.app.get("/api/model/")
        async def read_model():
            return self.get_model_graph()
            
        class ExpandNodeRequest(BaseModel):
            node: str

        @self.app.post("/api/model/expand")
        async def expand_node(request: ExpandNodeRequest):
            self.model_graph.nodes[request.node]['expanded'] = True
            return self.get_model_graph()
            
    def get_model_graph(self):
        graph = self.model_graph.copy()
        
        # Traverse the graph considering only edges with "edge_type" = "parent"
        # Start from all nodes that have 0 in-degree
        # Stop traversing when the node has 'expanded' = False
        def traverse_graph(graph: nx.DiGraph) -> list[str]:
            # Find all nodes with 0 in-degree
            zero_in_degree_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]
            visited = set()
            traversal_result = []

            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                traversal_result.append(node)
                if not graph.nodes[node].get('expanded', False):
                    return
                for neighbor in graph.successors(node):
                    if graph.edges[node, neighbor].get('edge_type') == 'parent':
                        dfs(neighbor)

            for node in zero_in_degree_nodes:
                dfs(node)
                
            return traversal_result
        
        # Remove all edges with attribute "edge_type" = "data_flow" from graph
        edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "data_flow"]
        graph.remove_edges_from(edges_to_remove)

        traversal_result = traverse_graph(graph)
        # Create a new graph with only the traversal_result nodes
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from([(node, data) for node, data in self.model_graph.nodes(data=True) if node in traversal_result])
        for u, v, d in self.model_graph.edges(data=True):
            if d.get("edge_type") == "data_flow" and u in traversal_result and v in traversal_result:
                new_graph.add_edge(u, v, **d)
                
        graph = new_graph
                
        nodes = [
            {
                **{k: v for k, v in data.items() if k not in ["output_tensor"]},
                "id": node,
            }
            for node, data in graph.nodes(data=True)
        ]
        edges = [
            {
                **{k: v for k, v in data.items()},
                "source": source,
                "target": target,
            }
            for source, target, data in graph.edges(data=True)
        ]
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
