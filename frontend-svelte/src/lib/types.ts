import type { Node, Edge } from "@xyflow/svelte";

export type ModelGraph = {
    nodes: Node[];
    edges: Edge[];
};

// export type Node = {
//     id: string;
//     name: string;
//     layer_type: string;
//     tensor_type: string;
//     input_shape: string | number[] | number[][] | null;
//     output_shape: string | number[] | number[][] | null;
//     is_leaf: boolean;
// };

// export type NodeWithLayout = Node & {
//     layout_horizontal: boolean;
//     tutorial_node: boolean;
//     position: { x: number; y: number; };
// };

// export type Edge = {
//     source: Node['id'];
//     target: Node['id'];
// };