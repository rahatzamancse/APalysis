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
        return f"{parent_name}->{module_name}"
    return module_name
# Hook function to add nodes and edges to the graph

def add_to_graph(G: nx.MultiDiGraph, module: torch.nn.Module | torch.nn.ModuleList | torch.nn.Sequential | torch.nn.ModuleDict, layer_name: str, output: torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor], parent_name: str | None, previous_layer_name: str | None) -> None:
    """
    Convert a PyTorch model to a networkx graph representation.

    Args:
    - model (torch.nn.Module): The PyTorch model to convert.
    - input_tensor (Union[torch.Tensor, dict, transformers.tokenization_utils_base.BatchEncoding]): A sample input tensor to perform a forward pass and generate outputs.

    Returns:
    - nx.DiGraph: A directed graph representing the model with nodes as layers and edges as data flow between layers.
    """
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
        output=output,
        output_type=output_type,
        output_shape=output_shape,
        layer_type=layer_type,
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
    G = nx.MultiDiGraph()
    
    # Function to recursively register hooks on all layers
    def register_hooks(module: torch.nn.Module | torch.nn.ModuleList | torch.nn.Sequential | torch.nn.ModuleDict, module_name: str, parent_name: str | None, previous_layer_name: str | None = None) -> None:
        
        def hook_fn(module, input, output):
            print(f"Hook called for module: {module_name}")
            add_to_graph(G, module, module_name, output, parent_name, previous_layer_name)

        module.register_forward_hook(hook_fn)
        this_previous_layer_name = previous_layer_name if previous_layer_name is None else previous_layer_name[:]
        for i, (name, child_module) in enumerate(module.named_children()):
            current_child_name = get_layer_name(name, module_name)
            register_hooks(child_module, current_child_name, module_name, (module_name if i == 0 else this_previous_layer_name))
            this_previous_layer_name = current_child_name

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
    
    G_hierarchy = G.copy()
    edges_to_remove = [(u, v) for u, v, d in G_hierarchy.edges(data=True) if d.get("edge_type") == "data_flow"]
    G_hierarchy.remove_edges_from(edges_to_remove)

    G_dataflow = G.copy()
    edges_to_remove = [(u, v) for u, v, d in G_dataflow.edges(data=True) if d.get("edge_type") == "parent"]
    G_dataflow.remove_edges_from(edges_to_remove)

    for node in G_hierarchy.nodes():
        if G_hierarchy.out_degree(node) != 0:
            child_nodes = [child for child in G_hierarchy.successors(node)]
            for child in child_nodes:
                if G_dataflow.out_degree(child) == 0:
                    G.add_edge(child, node, edge_type="data_flow")
                else:
                    next_child_nodes = [v for u, v, d in G_dataflow.edges(child, data=True)]
                    if all(G_hierarchy.has_edge(child, next_child) for next_child in next_child_nodes):
                        G.add_edge(child, node, edge_type="data_flow")
                        
    

    return G
