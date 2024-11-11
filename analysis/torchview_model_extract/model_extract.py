from typing import Sequence, Any, Mapping, Union, Iterable, Optional, List
from Graph import NewComputationGraph, Graph

import graphviz
import torch
from torch.jit import ScriptModule
import torchvision.models as models

from torchview.torchview import forward_prop

from torchview.torchview import process_input
import torchvision.transforms as transforms
import torchvision
import numpy as np


COMPILED_MODULES = (ScriptModule,)

INPUT_DATA_TYPE = Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]
CORRECTED_INPUT_DATA_TYPE = Optional[Union[Iterable[Any], Mapping[Any, Any]]]
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]
CORRECTED_INPUT_SIZE_TYPE = List[Union[Sequence[Any], torch.Size]]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inceptionv3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
imagenette = torchvision.datasets.ImageFolder('/home/insane/U/apalysis-evaluation/dataset-5')
sampled_images = torch.utils.data.Subset(imagenette, np.random.choice(len(imagenette), 10, replace=False).tolist())
transformed_images = [transform(image) for image, _ in sampled_images]
input_inceptionv3 = torch.stack(transformed_images)


class ComplexNetWithBranch(torch.nn.Module):
    def __init__(self):
        super(ComplexNetWithBranch, self).__init__()
        
        # Common layers before branching
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU()
        )
        
        self.inner_seq_1 = torch.nn.Sequential(
            torch.nn.Linear(15, 15),
            torch.nn.ReLU()
        )
        self.inner_seq_2 = torch.nn.Sequential(
            torch.nn.Linear(15, 15),
            torch.nn.ReLU()
        )
        
        # Branch 1
        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(20, 15),
            self.inner_seq_1,
            self.inner_seq_2
        )
        
        # Branch 2
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(20, 15),
            torch.nn.ReLU()
        )
        
        # Common layers after recombining
        self.fc_combine = torch.nn.Linear(30, 10)
        self.fc_output = torch.nn.Linear(10, 1)
        
        # ModuleList after recombination
        self.module_list = torch.nn.ModuleList([
            torch.nn.Linear(1, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ])

    def forward(self, x):
        # Common forward pass before branching
        x = self.seq(x)
        
        # Branching
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        
        # Concatenate the outputs of both branches
        x = torch.cat([branch1_out, branch2_out], dim=1)
        
        # Pass through the combined layer
        x = self.fc_combine(x)
        x = self.fc_output(x)
        
        # ModuleList forward pass
        for layer in self.module_list:
            x = layer(x)
        
        return x

myModel = ComplexNetWithBranch()




model_mode = "eval"
# MODEL, INPUT = inceptionv3, input_inceptionv3
MODEL, INPUT = myModel, torch.randn(1, 10)

def get_model_graph(model: torch.nn.Module, inputs: list[torch.Tensor]) -> Graph:
    visual_graph = graphviz.Digraph(name='model', engine='dot')
    input_recorder_tensor, kwargs_record_tensor, input_nodes = process_input(
        inputs, None, {}, device
    )
    show_shapes = True
    expand_nested = True
    hide_inner_tensors = False
    hide_module_functions = False
    roll = True
    depth = 5
    computation_graph = NewComputationGraph(
        visual_graph, input_nodes, show_shapes, expand_nested,
        hide_inner_tensors, hide_module_functions, roll, depth
    )

    forward_prop(
        model, input_recorder_tensor, device, computation_graph,
        model_mode, **kwargs_record_tensor
    )

    computation_graph.fill_visual_graph()

    return computation_graph.graph
