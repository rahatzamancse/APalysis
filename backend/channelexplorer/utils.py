import graphviz
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchview.torchview import forward_prop, process_input

from .Graph import Graph, NewComputationGraph

IMAGE_TYPE = np.ndarray # [channel, width, height]
GRAY_IMAGE_TYPE = np.ndarray # [width, height]

def get_activation_overlay(input_img: IMAGE_TYPE, activation: GRAY_IMAGE_TYPE, cmap=plt.cm.jet, alpha=0.3) -> IMAGE_TYPE:
    """Draws the activation overlay on the input image with the given colormap and alpha transparency

    Args:
        input_img (IMAGE_TYPE): The input image
        activation (GRAY_IMAGE_TYPE): The activation image
        cmap (str, optional): The color map. Defaults to plt.cm.jet.
        alpha (float, optional): The alpha used for blending the images. Defaults to 0.3.

    Returns:
        IMAGE_TYPE: The final image of the shape of input_img
    """
    act_img = Image.fromarray(activation)
    act_img = act_img.resize((input_img.shape[1], input_img.shape[0]), Image.BILINEAR)
    act_img = np.array(act_img)
    # normalize act_img
    act_img = (act_img - act_img.min()) / (act_img.max() - act_img.min())
    act_rgb = cmap(act_img)
    
    # normalize input_img
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    
    # convert to rgb if input_img is grayscale
    if input_img.ndim == 2:
        input_img = np.repeat(input_img[...,None], 3, axis=2)

    # Blend act_img to original image
    out_img = np.zeros(input_img.shape, dtype=input_img.dtype)
    out_img[:,:,:] = ((1-alpha) * input_img[:,:,:] + alpha * act_rgb[:,:,:3]).astype(input_img.dtype)
    
    out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())
    return out_img


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
        hide_inner_tensors, hide_module_functions, roll, depth, model
    )

    forward_prop(
        model, input_recorder_tensor, device, computation_graph,
        "eval", **kwargs_record_tensor
    )

    computation_graph.fill_visual_graph()
    
    if save_dot:
        computation_graph.visual_graph.render(filename="model.dot")

    return computation_graph.graph


def single_activation_distance(activation1_summary: np.ndarray, activation2_summary: np.ndarray):
    return np.sum(np.abs(activation1_summary - activation2_summary))

def single_activation_jaccard_distance(activation1_summary: np.ndarray, activation2_summary: np.ndarray, threshold: float = 0.5):
    assert len(activation1_summary) == len(activation2_summary)
    activation1_summary = activation1_summary > threshold
    activation2_summary = activation2_summary > threshold
    size = len(activation1_summary)
    num_differences = sum(
        int(activation1_summary[i] != activation2_summary[i]) for i in range(size))
    distance = num_differences / size
    return distance


def activation_distance(activation1_summary: dict[str, np.ndarray], activation2_summary: dict[str, np.ndarray]):
    dist = 0
    for act1, act2 in zip(activation1_summary.values(), activation2_summary.values()):
        dist += single_activation_distance(act1, act2)
    return dist


def rescale_img(img):
    img = img.squeeze()
    img = img - img.min()
    img = img / img.max()
    img *= 255
    img = img.astype(np.uint8)
    return img

def _find_optimal_k(X, max_k=10, method='elbow'):
    """
    Find the optimal number of clusters (K) using the Elbow Method or Silhouette Score.
    
    Parameters:
    - X: ndarray, the data points in n-dimensions.
    - max_k: int, the maximum number of clusters to try (default=10).
    - method: str, either 'elbow' (default) or 'silhouette' to determine the optimal K.
    
    Returns:
    - int, the optimal number of clusters (K).
    """
    # List to store inertia values for each K
    inertia = []
    silhouette_scores = []
    
    max_k = min(max_k, len(X)-1)

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        
        # Inertia (sum of squared distances to nearest cluster center)
        inertia.append(kmeans.inertia_)
        
        # Silhouette score (used if the method is silhouette)
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    if method == 'elbow':
        # Find the elbow point by checking the curvature
        deltas = np.diff(inertia)
        second_deltas = np.diff(deltas)
        optimal_k = np.argmin(second_deltas) + 2  # +2 because index 0 corresponds to k=2
        
    elif method == 'silhouette':
        # Find the maximum silhouette score
        optimal_k = np.argmax(silhouette_scores) + 2  # +2 because k starts at 2
    
    return optimal_k
