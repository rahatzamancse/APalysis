export type ModelGraph = {
    nodes: (FunctionNode | TensorNode | ContainerNode)[];
    edges: LayerEdge[];
};

export interface BaseNode {
    id: number;
    name: string;
}

export interface FunctionNode extends BaseNode {
    input_shape: number[] | number[][];
    output_shape: number[] | number[][];
    node_type: 'function'
};

export interface TensorNode extends BaseNode {
    value: number[] | number[][] | null;
    node_type: 'tensor'
}

export interface ContainerNode extends BaseNode {
    children: string[];
    node_type: 'container'
}

export interface LayerEdge {
    source: BaseNode['id'];
    target: BaseNode['id'];
    label: string;
};

export type LayerNode = TensorNode | FunctionNode | ContainerNode;