import { type Node, type Edge } from '@xyflow/svelte';

export function createGraph() {
    let nodes = $state<Node[]>([{ id: '1', position: { x: 0, y: 0 }, data: { label: 'Loading Graph\nPlease Wait' } }]);
    let edges = $state<Edge[]>([]);

    return {
        get nodes() {
            return nodes;
        },
        get edges() {
            return edges;
        },
        setNodes: (newNodes: Node[]) => {
            nodes = newNodes;
        },
        setEdges: (newEdges: Edge[]) => {
            edges = newEdges;
        }
    }
}