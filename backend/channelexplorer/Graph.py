import numpy as np
from typing import List, override
import graphviz
from torchview import TensorNode, FunctionNode, ModuleNode
from torchview.computation_graph import ComputationGraph, COMPUTATION_NODES

type NodeID = int

class GraphNode:
    def __init__(self, id: NodeID, label: str):
        self.id = id
        self.label = label
        
    def __repr__(self) -> str:
        return f'{self.id}: {self.label}'
            
class GraphTensorNode(GraphNode):
    def __init__(self, id: NodeID, label: str, value: np.ndarray):
        super().__init__(id, label)
        self.value = value
        
    def get_shape(self) -> tuple[int, ...]:
        return self.value.shape
        
    def __repr__(self) -> str:
        return f'{self.id}: Tensor {self.label} {self.value.shape}'

class GraphContainerNode(GraphNode):
    def __init__(self, id: NodeID, label: str, children: List[NodeID]):
        super().__init__(id, label)
        self.children: List[NodeID] = children
        
    def __repr__(self) -> str:
        return f'{self.id}: Container {self.label}'
        
class GraphFunctionNode(GraphNode):
    def __init__(self, id: NodeID, label: str, input_shape: list[tuple[int, ...]], output_shape: list[tuple[int, ...]], weights: list[np.ndarray]):
        super().__init__(id, label)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = weights
        
    def __repr__(self) -> str:
        return f'{self.id}: Function {self.label} {self.input_shape} -> {self.output_shape}'

class GraphEdge:
    def __init__(self, source: NodeID, target: NodeID, label: str | None = None):
        self.source = source
        self.target = target
        self.label = label
        
    def __repr__(self) -> str:
        return f'{self.source} -> {self.target} {self.label}'
    
class Graph:
    def __init__(self, nodes: List[GraphNode] = [], edges: List[GraphEdge] = []):
        self.nodes = nodes
        self.edges = edges
        
    def add_node(self, node: GraphNode) -> None:
        self.nodes.append(node)
        
    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)
        
    def __repr__(self) -> str:
        ret = "Graph"
        for node in self.nodes:
            ret += f"\n\t{node}"
        for edge in self.edges:
            ret += f"\n\t{edge}"
        ret += "\n"
        return ret

class NewComputationGraph(ComputationGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = Graph()
        
    @override
    def add_node(self, node: COMPUTATION_NODES, subgraph: graphviz.Digraph | None = None) -> None:
        '''Adds node to the graphviz with correct id, label and color
        settings. Updates state of running_node_id if node is not
        identified before. Also populates the graph'''
        super().add_node(node, subgraph)
        
        node_id = self.id_dict[node.node_id]
        label = node.name
        # add to this graph
        if isinstance(node, TensorNode):
            self.graph.add_node(GraphTensorNode(node_id, label, node.tensor_data))
        elif isinstance(node, FunctionNode):
            # TODO: get weights
            self.graph.add_node(GraphFunctionNode(node_id, label, node.input_shape, node.output_shape, []))
        elif isinstance(node, ModuleNode):
            self.graph.add_node(GraphContainerNode(node_id, label, [child.node_id for child in node.children]))
        
        if subgraph:
            # check if subgraph already exists
            subgraph_node = next((node for node in self.graph.nodes if isinstance(node, GraphContainerNode) and node.id == subgraph.name), None)
            if not subgraph_node:
                self.graph.add_node(GraphContainerNode(subgraph.name, subgraph.name, [node_id]))
            else:
                subgraph_node.children.append(node_id)
            
    @override
    def add_edge( self, edge_ids: tuple[int, int], edg_cnt: int) -> None:
        super().add_edge(edge_ids, edg_cnt)

        tail_id, head_id = edge_ids
        label = None if edg_cnt == 1 else f' x{edg_cnt}'
        self.graph.add_edge(GraphEdge(tail_id, head_id, label))
        
