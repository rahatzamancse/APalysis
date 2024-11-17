export type ModelGraph = {
    nodes: (FunctionNode | TensorNode | ContainerNode)[];
    edges: LayerEdge[];
};

export interface BaseNode {
    id: string;
    name: string;
    addWindow?: (from: BaseNode['id'], newNodeData: TensorWindowData | FunctionWindowData, width?: number, height?: number) => void;
}

export interface FunctionNode extends BaseNode {
    input_shape: number[] | number[][];
    output_shape: number[] | number[][];
    node_type: 'function'
};

export interface TensorNode extends BaseNode {
    node_type: 'tensor',
    shape: number[] | number[][];
}

export interface ContainerNode extends BaseNode {
    children: BaseNode['id'][];
    node_type: 'container'
}

export interface LayerEdge {
    source: BaseNode['id'];
    target: BaseNode['id'];
    label: string;
};

export type LayerNode = TensorNode | FunctionNode | ContainerNode;

export interface WindowNode {
    id: BaseNode['id'];
    fromId: BaseNode['id'];
    name: string;
}
export interface TensorWindowData extends WindowNode {
    type: 'tensorWindow';
    tensorId: string;
    shape: number[] | number[][];
}

export interface FunctionWindowData extends WindowNode {
    input_shape: number[] | number[][];
    output_shape: number[] | number[][];
    type: 'functionWindow';
}