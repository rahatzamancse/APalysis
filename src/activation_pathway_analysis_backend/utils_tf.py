import numpy as np
from typing import Dict, Any, Literal
from ast import literal_eval
import re
import tensorflow as tf
import tensorflow.keras as K
import networkx as nx
from .types import NodeInfo, IMAGE_TYPE, GRAY_IMAGE_TYPE
from .utils import *

# def get_example(ds, count=1) -> tuple[np.ndarray, int]:
#     if count == 1:
#         return next(ds.shuffle(10).take(count).as_numpy_iterator())
#     else:
#         return list(ds.shuffle(10).take(count).as_numpy_iterator())


def parse_model_graph(model: K.Model, layers_to_show: Literal["all"]|list[str] = 'all') -> Dict[str, Any]:
    activation_pathway_full = tensorflow_model_to_graph(model)
    simple_activation_pathway_full = remove_intermediate_node(
        activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['layer_type'] not in ['Conv2D', 'Dense', 'InputLayer', 'Concatenate'])
    
    if layers_to_show != 'all':
        simple_activation_pathway_full = remove_intermediate_node(
            simple_activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['name'] not in layers_to_show)

    node_pos = get_model_layout(simple_activation_pathway_full)

    # normalize node positions to be between 0 and 1
    min_x = min(pos[0] for pos in node_pos.values())
    max_x = max(pos[0] for pos in node_pos.values())
    min_y = min(pos[1] for pos in node_pos.values())
    max_y = max(pos[1] for pos in node_pos.values())

    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1

    node_pos = {
        node: ((pos[0] - min_x)/(max_x - min_x), (pos[1] - min_y)/(max_y - min_y)) for node, pos in node_pos.items()
    }

    for node, pos in node_pos.items():
        simple_activation_pathway_full.nodes[node]['pos'] = {
            'x': pos[0], 'y': pos[1]}
    
    # Edge weights
    kernel_norms = {}
    for layer_id, layer_data in simple_activation_pathway_full.nodes(data=True):
        if layer_data['layer_type'] in ['Concatenate', 'InputLayer', 'Dense']:
            continue
        layer_name = layer_data['name']
        kernel = model.get_layer(layer_name).get_weights()
        kernel_norm = np.linalg.norm(kernel[0], axis=(0,1), ord=2).sum(axis=0)
        kernel_norms[layer_name] = kernel_norm.tolist()
        # kernel = ((kernel - kernel.min()) / (kernel.max() -
        #           kernel.min()) * 255).astype(np.uint8)
        # img = Image.fromarray(kernel)
        
    return {
        'graph': nx.node_link_data(simple_activation_pathway_full),
        'meta': {
            'depth': max(nx.shortest_path_length(simple_activation_pathway_full, next(n for n, d in simple_activation_pathway_full.nodes(data=True) if d['layer_type'] == 'InputLayer')).values())
        },
        'edge_weights': kernel_norms,
    }
    
def shuffle_or_noshuffle(dataset: tf.data.Dataset, shuffle: bool = False):
    """Shuffles the dataset if shuffle is True

    Args:
        dataset (tf.data.Dataset): The dataset
        shuffle (bool, optional): Whether to shuffle or not. Defaults to False.

    Returns:
        tf.data.Dataset: The shuffled or unshuffled dataset
    """
    if shuffle:
        # TODO: make the shuffling for whole data instead of first 10000 data
        # https://stackoverflow.com/questions/44792761/how-can-i-shuffle-a-whole-dataset-with-tensorflow
        # get the seed
        seed = np.random.randint(0, 1000)
        print("Shuffling with seed", seed)
        return dataset.shuffle(buffer_size=10000, seed=seed)
    else:
        return dataset

def parse_tensorflow_dot_label(label: str) -> NodeInfo:
    """Parses the label of the layer in tensorflow.keras.utils.model_to_dot

    Args:
        label (str): The label of the layer in tensorflow.keras.utils.model_to_dot

    Returns:
        NodeInfo: The parsed result.
    """
    pattern = re.compile(r"\{([\w\d_]+)\|(\{[\w\d_]+\|[\w\d_]+\}|[\w\d_]+)\|([\w\d_]+)\}\|\{input:\|output:\}\|\{\{([\[\]\(\),\w\d_ ]*)\}\|\{([\[\]\(\),\w\d_ ]*)\}\}")
    match = pattern.findall(label)
    name, layer_type, tensor_type, input_shape, output_shape = match[-1]
    layer_activation = None
    if '|' in layer_type:
        layer_type, layer_activation = layer_type.split('|')
        layer_type = layer_type[1:]
        layer_activation = layer_activation[:-1]
    ret: NodeInfo = {
        'name': name,
        'layer_type': layer_type,
        'tensor_type': tensor_type,
        'input_shape': literal_eval(input_shape),
        'output_shape': literal_eval(output_shape),
        'layer_activation': layer_activation,
        'kernel_size': None
    }
    return ret

def tensorflow_model_to_graph(model: K.Model) -> nx.Graph:
    """Converts a tensorflow.keras model to a networkx graph

    Args:
        model (K.Model): The model

    Returns:
        nx.Graph: The networkx graph
    """
    dot = K.utils.model_to_dot(
        model,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96,
        subgraph=True,
        layer_range=None,
        show_layer_activations=True
    )
    G = nx.nx_pydot.from_pydot(dot)

    all_node_info = {}
    for node, node_data in G.nodes(data=True):
        node_info = parse_tensorflow_dot_label(node_data['label'])
        
        # Get kernel size
        if node_info['layer_type'] == 'Conv2D':
            node_info['kernel_size'] = model.get_layer(node_info['name']).kernel_size
        elif node_info['layer_type'] == 'MaxPooling2D':
            node_info['kernel_size'] = model.get_layer(node_info['name']).pool_size
        elif node_info['layer_type'] == 'Dense':
            node_info['kernel_size'] = model.get_layer(node_info['name']).units
        else:
            node_info['kernel_size'] = ()

        all_node_info[node] = node_info

    nx.set_node_attributes(G, all_node_info)
    return G

def get_mask_activation_channels(mask_img, activations, summary_fn_image, threshold_fn=lambda layer, channel: channel > np.percentile(layer, 99)):
    masked_activations = {layer:[] for layer in activations}
    activated_channels = {layer:[] for layer in masked_activations}
    for layer, val in activations.items():
        mask = tf.image.resize(mask_img[:,:,np.newaxis], val.shape[1:3], method=tf.image.ResizeMethod.BILINEAR).numpy().squeeze()
        for channel_i, channel in enumerate(val[0].transpose(2,0,1)):
            masked_val = apply_mask(channel, mask)
            masked_val_summary = summary_fn_image(masked_val[np.newaxis,:,:,np.newaxis]).squeeze()
            masked_activations[layer].append(masked_val_summary.item())
        
        for channel_i, channel in enumerate(masked_activations[layer]):
            if threshold_fn(masked_activations[layer], channel):
                activated_channels[layer].append(channel_i)
    
#     return masked_activations
    return activated_channels
