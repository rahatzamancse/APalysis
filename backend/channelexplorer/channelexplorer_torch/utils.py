import torch
from ..Graph import NewComputationGraph, Graph
from torchview.torchview import process_input, forward_prop
import graphviz

def get_model_graph(model: torch.nn.Module, inputs: list[torch.Tensor], device: str = "cpu", save_dot: bool = False) -> Graph:
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
        "eval", **kwargs_record_tensor
    )

    computation_graph.fill_visual_graph()
    
    if save_dot:
        computation_graph.visual_graph.render(filename="model.dot")

    return computation_graph.graph
