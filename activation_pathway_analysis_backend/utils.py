from itertools import combinations
import json
import numpy as np
from typing import TypedDict, Optional
from ast import literal_eval
import re
import tensorflow as tf
import networkx as nx
from grandalf.layouts import SugiyamaLayout
import grandalf
from matplotlib import pyplot as plt
from PIL import Image

def get_example(ds, count=1) -> tuple[np.ndarray, int]:
    if count == 1:
        return next(ds.shuffle(10).take(count).as_numpy_iterator())
    else:
        return list(ds.shuffle(10).take(count).as_numpy_iterator())
    
class NodeInfo(TypedDict):
    name: str
    layer_type: str
    tensor_type: str
    input_shape: list|tuple
    output_shape: list|tuple
    layer_activation: Optional[str]
    kernel_size: Optional[list|tuple]

def parse_dot_label(label: str) -> NodeInfo:
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

def get_activation_overlay(input_img, activation, cmap=plt.cm.jet, alpha=0.3):
    act_img = Image.fromarray(activation)
    act_img = act_img.resize((input_img.shape[1], input_img.shape[0]), Image.BILINEAR)
    act_img = np.array(act_img)
    act_rgb = cmap(act_img)
    
    # convert to rgb if input_img is grayscale
    if input_img.ndim == 2:
        input_img = np.repeat(input_img[...,None], 3, axis=2)

    # Blend act_img to original image
    out_img = np.zeros(input_img.shape, dtype=input_img.dtype)
    out_img[:,:,:] = ((1-alpha) * input_img[:,:,:] + alpha * act_rgb[:,:,:3]).astype(input_img.dtype)
    return out_img

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

def preprocess(img_batch_with_label, size=[299,299]) -> tf.Tensor:
    img_batch, labels = img_batch_with_label
    img = tf.image.central_crop(img_batch, central_fraction=1)
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


def get_model_layout(G):
    # TODO: Work on getting a better layout

    # Naive layout
    # pos = {}
    # input_node, _ = next(node for node in G.nodes(data=True) if node[1]['layer_type'] == 'InputLayer')
    # G.nodes[input_node]['level'] = 0
    # tree_depth = 0
    # for node in nx.topological_sort(G):
    #     if G.nodes[node]['layer_type'] == 'InputLayer':
    #         continue
    #     level = max(
    #         G.nodes[predecessor]['level'] 
    #         for predecessor in G.predecessors(node)
    #     ) + 1
    #     G.nodes[node]['level'] = level
    #     tree_depth = max(tree_depth, level)
    # pos = nx.multipartite_layout(G, subset_key="level", align='horizontal', scale=-1)


    # Get nodes by level
    # nodes_by_level = [[] for tree_depth in range(tree_depth + 1)]

    # for node, node_data in simple_activation_pathway_full.nodes(data=True):
    #     nodes_by_level[node_data['level']].append(node)

    # [[nx.get_node_attributes(simple_activation_pathway_full, 'name')[node] for node in nodes] for nodes in nodes_by_level]


    # Sugiyama Layout from grandalf library
    g = grandalf.utils.convert_nextworkx_graph_to_grandalf(G)
    for v in g.V(): v.view = type("defaultview", (object,), {"w": 10, "h": 10})
    sug = grandalf.layouts.SugiyamaLayout(g.C[0])
    sug.init_all()
    sug.draw() # This only calculated the positions for each node.
    pos = {v.data: (v.view.xy[0], -v.view.xy[1]) for v in g.C[0].sV} # Extracts the positions
    return pos
