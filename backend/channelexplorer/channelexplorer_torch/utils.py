import torch
import networkx as nx
import transformers
from ..utils import unify_graph

def extract_activations_graphs(models, inputs):
    graphs = []
    for model, input in zip(models, inputs):
        graphs.append(extract_activations_graph(model, input))
    union_graph = unify_graph(graphs)
    return union_graph
    
# Function to create unique layer names
def get_layer_name(module_name: str, parent_name: str = "") -> str:
    if parent_name:
        return f"{parent_name}_{module_name}"
    return module_name
# Hook function to add nodes and edges to the graph

def add_to_graph(G: nx.DiGraph, module: torch.nn.Module | torch.nn.ModuleList | torch.nn.Sequential | torch.nn.ModuleDict, layer_name: str, output: torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor], parent_name: str | None, previous_layer_name: str | None) -> None:
    # Get the layer type, dtype, and shape
    layer_type = module.__class__.__name__
    if isinstance(output, tuple):
        output = output[0]
    elif isinstance(output, dict):
        if 'last_hidden_state' in output:
            output = output['last_hidden_state']
        elif 'logits' in output:
            output = output['logits']
        else:
            raise ValueError(f"Unknown output type: {output}")
    output_type = str(output.dtype)
    output_shape = tuple(output.shape)

    # Add node with attributes
    G.add_node(
        layer_name,
        name=layer_name,
        layer_type=layer_type,
        tensor_type=output_type,
        input_shape=tuple(module.input_shape) if hasattr(module, 'input_shape') else None,
        output_shape=output_shape,
        output_tensor=output,
    )

    # Add edge from the previous layer to the current layer if it's not the input layer
    if previous_layer_name:
        G.add_edge(previous_layer_name, layer_name, edge_type="data_flow")
    
    if parent_name:
        G.add_edge(parent_name, layer_name, edge_type="parent")


def extract_activations_graph(model: torch.nn.Module | torch.nn.ModuleList | torch.nn.Sequential | torch.nn.ModuleDict, input_tensor: torch.Tensor | dict | transformers.tokenization_utils_base.BatchEncoding) -> nx.DiGraph:
    """
    Convert a PyTorch model to a networkx graph representation.

    Args:
    - model (torch.nn.Module): The PyTorch model to convert.
    - input_tensor (Union[torch.Tensor, dict, transformers.tokenization_utils_base.BatchEncoding]): A sample input tensor to perform a forward pass and generate outputs.

    Returns:
    - nx.DiGraph: A directed graph representing the model with nodes as layers and edges as data flow between layers.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Function to recursively register hooks on all layers
    def register_hooks(module: torch.nn.Module | torch.nn.ModuleList | torch.nn.Sequential | torch.nn.ModuleDict, module_name: str, parent_name: str | None, previous_layer_name: str | None = None) -> None:
        module.register_forward_hook(lambda module, input, output: add_to_graph(G, module, module_name, output, parent_name, previous_layer_name))
        for name, child_module in module.named_children():
            current_child_name = get_layer_name(name, module_name)
            register_hooks(child_module, current_child_name, module_name, previous_layer_name)
            previous_layer_name = current_child_name

    # Register hooks starting from the root model
    register_hooks(model, type(model).__name__, None)

    try:
        # Perform a forward pass with the input tensor to populate the graph
        with torch.no_grad():
            if isinstance(input_tensor, (dict, transformers.tokenization_utils_base.BatchEncoding)):
                model(**input_tensor)
            else:
                model(input_tensor)
    finally:
        # remove all hooks
        for module in model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()

    return G