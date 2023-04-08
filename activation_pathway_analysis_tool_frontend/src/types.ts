export type ModelGraph = {
    nodes: Node[];
    edges: Edge[];
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
    pos?: { x: number; y: number; };
    layout_horizontal?: boolean;
};

export type Edge = {
    source: string;
    target: string;
};

export type Prediction = {
    prediction: string
}