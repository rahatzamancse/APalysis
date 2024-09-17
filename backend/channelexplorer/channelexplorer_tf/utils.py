import numpy as np
from typing import Dict, Any, Literal
import tensorflow as tf
import networkx as nx
from ..types import NodeInfo
from ..utils import *
import transformers


def extract_activations_graphs(models, inputs):
    graphs = []
    for model, input in zip(models, inputs):
        graphs.append(extract_activations_graph(model, input))
    union_graph = unify_graph(graphs)
    return union_graph

# Function to extract activations from self-attention, feed-forward, and hidden states
def get_transformer_block_activations(model, inputs):
    """
    Function to get intermediate activations from each transformer block, including self-attention and feed-forward activations.
    
    Args:
    - model: Transformer model from Hugging Face (e.g., GPT-2)
    - inputs: Tokenized input (output of tokenizer)
    
    Returns:
    - Dictionary containing activations for self-attention, feed-forward, and hidden states
    """
    activations = {}
    for i in inputs:
        # Run the model with output_hidden_states and output_attentions to get all intermediate data
        outputs = model(input_ids=i['input_ids'], output_hidden_states=True, output_attentions=True)

        # The hidden_states contains the hidden layer outputs, and attentions contains attention scores
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        activations_block = {
            "embedding_layer": hidden_states[0].numpy(),  # First hidden state is the embedding layer
        }

        # Iterate over the transformer blocks
        for i, (attention, hidden_state) in enumerate(zip(attentions, hidden_states[1:])):
            # Self-attention output (stored in attentions)
            activations_block[f"block_{i}_self_attention"] = attention.numpy()
            
            # Feed-forward output (this is typically captured in the hidden states after attention and feed-forward pass)
            activations_block[f"block_{i}_feed_forward"] = hidden_state.numpy()
        
        for k, v in activations_block.items():
            if k not in activations:
                activations[k] = []
            activations[k].append(v)
            
    return activations


def get_vit_block_activations(model, inputs):
    """
    Function to get intermediate activations from each transformer block in a ViT model, including self-attention and feed-forward activations.
    
    Args:
    - model: Vision Transformer model from Hugging Face.
    - inputs: Preprocessed image inputs (output of feature extractor).
    
    Returns:
    - Dictionary containing activations for self-attention, feed-forward, and hidden states from each block.
    """
    # Run the model with output_hidden_states and output_attentions to get all intermediate activations
    outputs = model(inputs, output_hidden_states=True, output_attentions=True)
    
    # Hidden states contain outputs from all layers; attentions contain attention maps
    logits = outputs.logits
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    
    activations = {
        "embedding_layer": hidden_states[0],  # First hidden state is the embedding layer
    }

    # Iterate over the transformer blocks in the ViT model
    for i, (attention, hidden_state) in enumerate(zip(attentions, hidden_states[1:])):
        # Self-attention output (stored in attentions)
        activations[f"block_{i}_self_attention"] = attention
        
        # Feed-forward output (captured in the hidden states)
        activations[f"block_{i}_feed_forward"] = hidden_state

    return activations


def extract_activations_graph(model, inputs):
    G = nx.DiGraph()
    last_layer_name = None
    last_output_tensor = inputs
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.InputLayer, )):
            G.add_node(
                layer.name,
                name=layer.name,
                layer_type=layer.__class__.__name__,
                tensor_type=str(layer.dtype),
                input_shape=layer.input_shape,
                output_shape=layer.output_shape,
                output_tensor=inputs,
                is_parent=False,
                parent=None,
            )
            last_layer_name = layer.name
            


        elif isinstance(layer, (transformers.TFGPT2MainLayer, transformers.TFGPT2LMHeadModel)):
            block_activations = get_transformer_block_activations(model, inputs)
            G.add_node(
                layer.name,
                name=layer.name,
                layer_type="TransformerBlock",
                config=layer.config.to_dict(),
                last_hidden_state_shape=tuple(layer.compute_output_shape(tf.TensorShape([None, model.config.n_positions]))['last_hidden_state']),
                past_key_values_shape=tuple(layer.compute_output_shape(tf.TensorShape([None, model.config.n_positions]))['past_key_values']),
                tensor_type=str(layer.dtype),
                parent=None,
                is_parent=True,
            )
            prev_block_name = None
            layer_type_map = {
                "embedding": "Embedding",
                "self_attention": "SelfAttention",
                "feed_forward": "FeedForward",
            }
            def get_layer_type(layer_name):
                for key, value in layer_type_map.items():
                    if key in layer_name:
                        return value
                return "Unknown"
            for layer_name, activation in block_activations.items():
                G.add_node(
                    layer_name,
                    name=layer_name,
                    layer_type=get_layer_type(layer_name),
                    output_shape=f"({activation[0].shape[0]}, InputLength, {activation[0].shape[2]})",
                    tensor_type=str(activation[0].dtype),
                    output_tensor=activation,
                    parent=layer.name,
                    is_parent=False,
                )
                if prev_block_name:
                    G.add_edge(prev_block_name, layer_name)
                prev_block_name = layer_name
                last_output_tensor = activation
                last_layer_name = layer_name
                
                
        elif isinstance(layer, transformers.models.vit.modeling_tf_vit.TFViTMainLayer):
            block_activations = get_vit_block_activations(model, inputs)
            G.add_node(
                layer.name,
                name=layer.name,
                layer_type="TransformerBlock",
                config=layer.config.to_dict(),
                tensor_type=str(layer.dtype),
                parent=None,
                is_parent=True,
            )
            prev_block_name = None
            layer_type_map = {
                "embedding": "Embedding",
                "self_attention": "SelfAttention",
                "feed_forward": "FeedForward",
            }
            def get_layer_type(layer_name):
                for key, value in layer_type_map.items():
                    if key in layer_name:
                        return value
                return "Unknown"
            for layer_name, activation in block_activations.items():
                G.add_node(
                    layer_name,
                    name=layer_name,
                    layer_type=get_layer_type(layer_name),
                    output_shape=tuple(activation.shape),
                    tensor_type=str(activation[0].dtype),
                    output_tensor=activation,
                    parent=layer.name,
                    is_parent=False,
                )
                if prev_block_name:
                    G.add_edge(prev_block_name, layer_name)
                prev_block_name = layer_name
                last_output_tensor = activation
                last_layer_name = layer_name



                
        elif isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.Concatenate,
                tf.keras.layers.MaxPooling2D,
                tf.keras.layers.BatchNormalization,
                tf.keras.layers.Activation,
                tf.keras.layers.AveragePooling2D,
                tf.keras.layers.GlobalAveragePooling2D,
                tf.keras.layers.GlobalMaxPooling2D,
            ),
        ):
            intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
            intermediate_output = intermediate_model(inputs)
            G.add_node(
                layer.name,
                name=layer.name,
                input_shape=layer.input_shape,
                layer_type=layer.__class__.__name__,
                output_shape=layer.output_shape,
                output_tensor=intermediate_output.numpy(),
                tensor_type=layer.dtype,
                is_parent=False,
                parent=None,
            )
            for inbound_node in layer._inbound_nodes:
                if isinstance(inbound_node.inbound_layers, list):
                    for inbound_layer in inbound_node.inbound_layers:
                        G.add_edge(inbound_layer.name, layer.name)
                else:
                    G.add_edge(inbound_node.inbound_layers.name, layer.name)
            last_output_tensor = intermediate_output
            last_layer_name = layer.name
            


                    
        elif isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Flatten)):
            intermediate_output = layer(last_output_tensor)
            G.add_node(
                layer.name,
                name=layer.name,
                layer_type=layer.__class__.__name__,
                input_shape=last_output_tensor.shape,
                output_shape=intermediate_output.shape,
                output_tensor=intermediate_output.numpy(),
                tensor_type=layer.dtype,
                is_parent=False,
                parent=None,
            )
            for inbound_node in layer._inbound_nodes:
                if isinstance(inbound_node.inbound_layers, list):
                    for inbound_layer in inbound_node.inbound_layers:
                        G.add_edge(inbound_layer.name, layer.name)
                else:
                    G.add_edge(inbound_node.inbound_layers.name, layer.name)
            else:
                G.add_edge(last_layer_name, layer.name)
            last_output_tensor = intermediate_output
            last_layer_name = layer.name
            


            
        else:
            print("Is a generic layer", layer.name, layer)
            raise Exception("Is a generic layer")
    return G

def parse_model_graphs(models: list[tf.keras.Model], layers_to_show: list[Literal["all"]|list[str]]) -> Dict[str, Any]:
    graphs = []
    for model_idx, (model, layers) in enumerate(zip(models, layers_to_show)):
        activation_pathway = tensorflow_model_to_graph(model)
    
        if layers != 'all':
            activation_pathway = remove_intermediate_node(
                activation_pathway,
                lambda node: \
                    not activation_pathway.nodes[node]['is_parent'] and \
                    activation_pathway.nodes[node]['name'] not in layers
            )
        
        # Remove duplicate edges
        new_activation_pathway = nx.DiGraph()
        ## copy nodes
        for node, node_data in activation_pathway.nodes(data=True):
            new_activation_pathway.add_node(node, **node_data)
        ## copy edges
        for edge in activation_pathway.edges:
            if new_activation_pathway.has_edge(edge[0], edge[1]):
                continue
            new_activation_pathway.add_edge(edge[0], edge[1])
        activation_pathway = new_activation_pathway
        
        # add model index
        for node in activation_pathway.nodes:
            activation_pathway.nodes[node]['model_idx'] = model_idx

        graphs.append(activation_pathway)
            
    unified_graph = unify_graph(graphs)
    
    node_link_data = nx.node_link_data(unified_graph)
    
    return {
        "graph": node_link_data,
    }
    
def tensorflow_model_to_graph(model: tf.keras.Model) -> nx.DiGraph:
    """Converts a tensorflow.keras model or a Hugging Face transformer model to a networkx graph

    Args:
        model (tf.keras.Model or transformers model): The model

    Returns:
        nx.DiGraph: The networkx graph
    """
    G = nx.DiGraph()
    
    def add_layer_to_graph(layer, parent_name=None):
        layer_info: NodeInfo = {
            'name': layer.name,
            'layer_type': layer.__class__.__name__,
            'tensor_type': str(layer.dtype),
            'input_shape': getattr(layer, 'input_shape', None),
            'output_shape': getattr(layer, 'output_shape', None),
            'layer_activation': getattr(layer, 'activation', None).__name__ if hasattr(layer, 'activation') and layer.activation else None,
            'kernel_size': None,
            'parent': None,
            'is_parent': hasattr(layer, 'layers')
        }
        
        # Get kernel size for specific layer types
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_info['kernel_size'] = layer.kernel_size
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            layer_info['kernel_size'] = layer.pool_size
        elif isinstance(layer, tf.keras.layers.Dense):
            layer_info['kernel_size'] = (layer.units,)
        
        G.add_node(layer.name, **layer_info)
        
        if parent_name:
            layer_info['parent'] = parent_name
        
        # Add edges from input layers to this layer
        if hasattr(layer, '_inbound_nodes'):
            for node in layer._inbound_nodes:
                if isinstance(node.inbound_layers, list):
                    for inbound_layer in node.inbound_layers:
                        G.add_edge(inbound_layer.name, layer.name)
                else:
                    inbound_layer = node.inbound_layers
                    G.add_edge(inbound_layer.name, layer.name)
        
        # Recursively add sublayers
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                add_layer_to_graph(sublayer, layer.name)
        
        # Special handling for GPT-2 model
        if isinstance(layer, (transformers.TFGPT2Model, transformers.TFGPT2LMHeadModel)):
            gpt2_model = layer.transformer if isinstance(layer, transformers.TFGPT2LMHeadModel) else layer
            prev_block_name = layer.name
            for i, block in enumerate(gpt2_model.h):
                block_name = f"{layer.name}_block_{i}"
                G.add_node(block_name, name=block_name, layer_type="TransformerBlock", is_parent=True)
                G.add_edge(prev_block_name, block_name)
                
                # Add attention and MLP layers
                attn_name = f"{block_name}_attention"
                mlp_name = f"{block_name}_mlp"
                G.add_node(attn_name, name=attn_name, layer_type="Attention", is_parent=False)
                G.add_node(mlp_name, name=mlp_name, layer_type="MLP", is_parent=False)
                G.add_edge(block_name, attn_name)
                G.add_edge(attn_name, mlp_name)
                
                prev_block_name = block_name
            
            # Add final layer norm
            ln_f_name = f"{layer.name}_ln_f"
            G.add_node(ln_f_name, name=ln_f_name, layer_type="LayerNorm", is_parent=False)
            G.add_edge(prev_block_name, ln_f_name)
    
    # Start the recursive process
    add_layer_to_graph(model)
    
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
