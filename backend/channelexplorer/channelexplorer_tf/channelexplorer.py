from ..channelexplorer import Server
from sklearn.cluster import KMeans
from typing import Literal, Callable, Any, Dict
from ..types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE, IMAGE_TYPE
from .. import metrics
from fastapi.concurrency import asynccontextmanager
from fastapi import FastAPI, File, Response
from fastapi.middleware.cors import CORSMiddleware
from . import utils as utils
import tensorflow as tf
from tensorflow import keras as K
import keract
import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)



class ChannelExplorer_TF(Server):
    def __init__(
        self,
        model: K.Model,
        summary_fn_image: Callable[
            [IMAGE_BATCH_TYPE], SUMMARY_BATCH_TYPE
        ] = metrics.summary_fn_image_l2,
        summary_fn_dense: Callable[
            [DENSE_BATCH_TYPE], SUMMARY_BATCH_TYPE
        ] = metrics.summary_fn_dense_identity,
        layers_to_show: list[str] | Literal["all"] = "all",
        log_level: Literal["info", "debug"] = "info",
    ):
        self.model = model
        self.log_level = log_level
        self.summary_fn_image = summary_fn_image
        self.summary_fn_dense = summary_fn_dense
        self.layers_to_show = layers_to_show

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

        @self.app.get("/api/model/")
        async def read_model():
            return self.model_graph
        
    
    def feed_inputs(self, inputs: list[np.ndarray] | np.ndarray):
        self.inputs = inputs
        
        self.activations = utils.get_activations(
            self.model,
            self.inputs,
            layers=self.layers_to_show,
        )
        
        self.activations_summary = {}
        for layer_name, layer_data in self.activations.items():
            if len(layer_data.shape) == 2:
                self.activations_summary[layer_name] = self.summary_fn_dense(layer_data)
            elif len(layer_data.shape) == 4:
                self.activations_summary[layer_name] = self.summary_fn_image(layer_data)
            else:
                raise ValueError(f"{layer_name} has an unsupported shape: {layer_data.shape}")
                

        logging.info(f"Done feeding inputs. Number of images: {len(inputs)}")