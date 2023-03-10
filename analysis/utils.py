from itertools import combinations
import json
import numpy as np
from typing import TypedDict, Optional
from ast import literal_eval
import re
import tensorflow as tf
import networkx as nx

with open('imagenet_class_index.json', 'r') as f:
    imagenet_inception_v3_labels: dict[str, list[str]] = json.load(f)
    
def get_example(ds, count=1) -> tuple[np.ndarray, int]:
    if count == 1:
        return next(ds.shuffle(10).take(count).as_numpy_iterator())
    else:
        return list(ds.shuffle(10).take(count).as_numpy_iterator())
    
def class_id_to_name(class_id):
    for imagenet_id, name in imagenet_inception_v3_labels.values():
        if imagenet_id == class_id:
            return name
    return "Not found"
        
def pred_to_name(pred):
    return imagenet_inception_v3_labels[str(pred)][1]

class NodeInfo(TypedDict):
    name: str
    layer_type: str
    tensor_type: str
    input_shape: list|tuple
    output_shape: list|tuple
    layer_activation: Optional[str]

def parse_dot_label(label) -> NodeInfo:
    pattern = re.compile(r"\{([\w\d_]+)\|(\{[\w\d_]+\|[\w\d_]+\}|[\w\d_]+)\|([\w\d_]+)\}\|\{input:\|output:\}\|\{\{([\[\]\(\),\w\d_ ]*)\}\|\{([\[\]\(\),\w\d_ ]*)\}\}")
    match = pattern.findall(label)
    name, layer_type, tensor_type, input_shape, output_shape = match[-1]
    if '|' in layer_type:
        layer_type, layer_activation = layer_type.split('|')
        layer_type = layer_type[1:]
        layer_activation = layer_activation[:-1]
    ret = {
        'name': name,
        'layer_type': layer_type,
        'tensor_type': tensor_type,
        'input_shape': literal_eval(input_shape),
        'output_shape': literal_eval(output_shape)
    }
    if '|' in layer_type:
        ret['layer_activation'] = layer_activation
    return ret


def model_to_graph(model):
    dot = tf.keras.utils.model_to_dot(
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
        node_info = parse_dot_label(node_data['label'])
        all_node_info[node] = node_info

    nx.set_node_attributes(G, all_node_info)
    return G

def preprocess(img_batch_with_label, size=[299,299]) -> tf.Tensor:
    img_batch, labels = img_batch_with_label
    img = tf.image.central_crop(img_batch, central_fraction=0.875)
    img = tf.image.resize(img, size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32) / 255
    img -= 0.5
    img *= 2.0
    return img, labels


def remove_intermediate_node(G: nx.Graph, node_removal_predicate: callable):
    '''
    Loop over the graph until all nodes that match the supplied predicate 
    have been removed and their incident edges fused.
    src: https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges
    '''
    g = G.copy()
    while any(node_removal_predicate(node) for node in g.nodes):

        g0 = g.copy()

        for node in g.nodes:
            if node_removal_predicate(node):

                if g.is_directed():
                    in_edges_containing_node = list(g0.in_edges(node))
                    out_edges_containing_node = list(g0.out_edges(node))

                    for in_src, _ in in_edges_containing_node:
                        for _, out_dst in out_edges_containing_node:
                            g0.add_edge(in_src, out_dst)
                            # dist = nx.shortest_path_length(
                            #   g0, in_src, out_dst, weight='weight'
                            # )
                            # g0.add_edge(in_src, out_dst, weight=dist)
                else:
                    edges_containing_node = g.edges(node)
                    dst_to_link = [e[1] for e in edges_containing_node]
                    dst_pairs_to_link = list(combinations(dst_to_link, r = 2))
                    for pair in dst_pairs_to_link:
                        g0.add_edge(pair[0], pair[1])
                        # dist = nx.shortest_path_length(
                        # g0, pair[0], pair[1], weight='weight'
                        # )
                        # g0.add_edge(pair[0], pair[1], weight=dist)
                
                g0.remove_node(node)
                break
        g = g0
    return g