export type ModelGraph = {
    nodes: Node[];
    edges: Edge[];
};

export type Node = {
    id: string;
    label: string;
    layer_type: string;
    name: string;
    input_shape: (null|number)[];
    output_shape: (null|number)[];
    tensor_type: string;
    pos?: { x: number; y: number; };
};

export type Edge = {
    source: string;
    target: string;
};

export type Prediction = {
    prediction: string
}