export type ModelGraph = {
    nodes: Node[];
    edges: Edge[];
};

export type Node = {
    id: string;
    label: string;
    layer_type: string;
    name: string;
    output_shape: string | number[] | number[][];
    tensor_type: string;
    is_parent: boolean;
    parent: string | null;
};

export type NodeWithLayout = Node & {
    layout_horizontal: boolean;
    tutorial_node: boolean;
    position: { x: number; y: number; };
};

export type Edge = {
    source: Node['id'];
    target: Node['id'];
};