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
    output_shape: [null|number, ...number[]];
    tensor_type: string;
    pos?: { x: number; y: number; };
    layout_horizontal?: boolean;
    tutorial_node?: boolean;
};

export interface Node_CNN extends Node {
    layer_type: "Conv2D";
    input_shape: [null, number, number, number];
    kernel_size: [number, number];
    output_shape: [null, number, number, number];
    tensor_type: "float32";
    out_edge_weight: number[];
}

export type Edge = {
    source: string;
    target: string;
};

export type Prediction = {
    prediction: string
}