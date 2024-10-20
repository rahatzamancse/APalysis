export type ModelGraph = {
    nodes: LayerNode[];
    edges: LayerEdge[];
};

export interface LayerNode {
    id: string;
    name: string;
    layer_type: string;
    tensor_type: string;
    output_shape: string | number[] | number[][] | null;
    is_leaf: boolean;
    expanded: boolean;
};

export interface LayerEdge {
    source: LayerNode['id'];
    target: LayerNode['id'];
    edge_type: 'parent' | 'data_flow';
};