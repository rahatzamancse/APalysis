from typing import Any, Callable, Literal

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from .Graph import GraphContainerNode, GraphFunctionNode, GraphTensorNode
from .metrics import summary_fn_dense_identity, summary_fn_image_l2
from .types import DENSE_BATCH_TYPE, IMAGE_BATCH_TYPE, SUMMARY_BATCH_TYPE
from .utils import get_model_graph


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

