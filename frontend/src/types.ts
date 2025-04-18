export type ModelGraph = {
    nodes: Node[];
    edges: Edge[];
    meta: {
        depth: number;
    };
};

export type Node = {
    id: string;
    label: string;
    layer_type: string;
    name: string;
    input_shape: [null|number, ...number[]];
    kernel_size: number[];
    output_shape: [null|number, ...number[]];
    tensor_type: string;
    out_edge_weight: number[];
    pos?: { x: number; y: number; };
    layout_horizontal?: boolean;
    tutorial_node?: boolean;
};

export type Edge = {
    source: string;
    target: string;
};

export type Prediction = {
    prediction: string
}

export interface ClusterNode {
    id: number;
    type: 'mainclass' | 'subclass';
    level: number;
    instanceIds: number[];
    x?: number;
    y?: number;
    r?: number;
} 