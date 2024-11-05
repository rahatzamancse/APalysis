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
import itertools
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn import manifold
from ..types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE, IMAGE_TYPE
from .. import metrics
import networkx as nx

import torch
from . import utils as utils
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
        _hierarchy_graph = self.model_graph.copy()
        edges_to_remove = [(u, v) for u, v, d in _hierarchy_graph.edges(data=True) if d.get("edge_type") == "data_flow"]
        _hierarchy_graph.remove_edges_from(edges_to_remove)
        for node in self.model_graph.nodes(data=True):
            is_leaf = _hierarchy_graph.out_degree(node[0]) == 0
            node[1]['is_leaf'] = is_leaf
            node[1]['expanded'] = is_leaf

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

        @self.app.post("/api/model/collapse")
        async def collapse_node(request: ExpandNodeRequest):
            self.model_graph.nodes[request.node]['expanded'] = False
            
            _hierarchy_graph = self.model_graph.copy()
            edges_to_remove = [(u, v) for u, v, d in _hierarchy_graph.edges(data=True) if d.get("edge_type") == "parent"]
            _hierarchy_graph.remove_edges_from(edges_to_remove)
            
            def collapse_children(node):
                for child in _hierarchy_graph.successors(node):
                    if not self.model_graph.nodes[child]['is_leaf']:
                        self.model_graph.nodes[child]['expanded'] = False
                        collapse_children(child)

            collapse_children(request.node)

            return self.get_model_graph()
            
    def get_model_graph(self):
        graph = self.model_graph.copy()
        hierarchy_graph = graph.copy()
        edges_to_remove = [(u, v) for u, v, d in hierarchy_graph.edges(data=True) if d.get("edge_type") == "data_flow"]
        hierarchy_graph.remove_edges_from(edges_to_remove)
        
        dataflow_graph = self.model_graph.copy()
        edges_to_remove = [(u, v) for u, v, d in dataflow_graph.edges(data=True) if d.get("edge_type") == "parent"]
        dataflow_graph.remove_edges_from(edges_to_remove)
        
        parent_nodes_to_redirect = []
        
        # Traverse the hierarchy graph starting from root nodes
        # Stop traversing when the node has 'expanded' = False or is a leaf node
        def traverse_graph(graph: nx.DiGraph, hierarchy_graph: nx.DiGraph, dataflow_graph: nx.DiGraph) -> list[str]:
            # Find all nodes with 0 in-degree
            zero_in_degree_nodes = [node for node, in_degree in hierarchy_graph.in_degree() if in_degree == 0]
            traversal_result = []

            def bfs(start_node):
                queue = [start_node]
                while queue:
                    node = queue.pop(0)
                    is_leaf = hierarchy_graph.nodes[node]['is_leaf']
                    is_expanded = hierarchy_graph.nodes[node]['expanded']
                    if is_leaf or not is_expanded:
                        traversal_result.append(node)
                    if not is_leaf and is_expanded:
                        for child in hierarchy_graph.successors(node):
                            queue.append(child)
                        parent_nodes_to_redirect.append(node)

            for node in zero_in_degree_nodes:
                bfs(node)
                
            return traversal_result

        traversal_result = traverse_graph(graph, hierarchy_graph, dataflow_graph)
        # Create a new graph with only the traversal_result nodes
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from([(node, data) for node, data in self.model_graph.nodes(data=True) if node in traversal_result])
        # Add all data_flow edges to new_graph from self.model_graph
        edges = [ (u, v, d) for u, v, d in self.model_graph.edges(data=True) if d.get("edge_type") == "data_flow" and u in traversal_result and v in traversal_result ]
        new_graph.add_edges_from(edges)
        graph = new_graph
        
        for node in parent_nodes_to_redirect:
            in_coming_nodes = [s for s in dataflow_graph.predecessors(node) if s not in (nx.descendants(hierarchy_graph, node) | {node}) and s not in (nx.ancestors(hierarchy_graph, node) | {node})]
            out_going_nodes = [s for s in dataflow_graph.successors(node) if s not in (nx.descendants(hierarchy_graph, node) | {node}) and s not in (nx.ancestors(hierarchy_graph, node) | {node})]
            
            incoming_children = [s for s in dataflow_graph.successors(node) if s in (nx.descendants(hierarchy_graph, node) | {node})]
            outgoing_children = [s for s in dataflow_graph.predecessors(node) if s in (nx.descendants(hierarchy_graph, node) | {node})]
            
            for u, v in itertools.product(in_coming_nodes, incoming_children):
                graph.add_edge(u, v, edge_type="data_flow")
            for u, v in itertools.product(outgoing_children, out_going_nodes):
                graph.add_edge(u, v, edge_type="data_flow")

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

