import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { cubicOut } from "svelte/easing";
import type { TransitionConfig } from "svelte/transition";
import dagre from '@dagrejs/dagre';
import type { TensorNode, LayerEdge, FunctionNode, ContainerNode, LayerNode } from '$lib/types';
import type { Node, Edge } from '@xyflow/svelte';

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

type FlyAndScaleParams = {
	y?: number;
	x?: number;
	start?: number;
	duration?: number;
};

export const flyAndScale = (
	node: Element,
	params: FlyAndScaleParams = { y: -8, x: 0, start: 0.95, duration: 150 }
): TransitionConfig => {
	const style = getComputedStyle(node);
	const transform = style.transform === "none" ? "" : style.transform;

	const scaleConversion = (
		valueA: number,
		scaleA: [number, number],
		scaleB: [number, number]
	) => {
		const [minA, maxA] = scaleA;
		const [minB, maxB] = scaleB;

		const percentage = (valueA - minA) / (maxA - minA);
		const valueB = percentage * (maxB - minB) + minB;

		return valueB;
	};

	const styleToString = (
		style: Record<string, number | string | undefined>
	): string => {
		return Object.keys(style).reduce((str, key) => {
			if (style[key] === undefined) return str;
			return str + `${key}:${style[key]};`;
		}, "");
	};

	return {
		duration: params.duration ?? 200,
		delay: 0,
		css: (t) => {
			const y = scaleConversion(t, [0, 1], [params.y ?? 5, 0]);
			const x = scaleConversion(t, [0, 1], [params.x ?? 0, 0]);
			const scale = scaleConversion(t, [0, 1], [params.start ?? 0.95, 1]);

			return styleToString({
				transform: `${transform} translate3d(${x}px, ${y}px, 0) scale(${scale})`,
				opacity: t
			});
		},
		easing: cubicOut
	};
};

// Build a dependency graph where each node points to its children and then performing a depth-first traversal to sort nodes
function sortNodesByParentDependency(nodes: Node[]): any[] {
	// Create a map to store nodes by their IDs for quick access
	const nodeMap = new Map(nodes.map(node => [node.id, node]));

	// Initialize adjacency list to build dependency graph
	const adjacencyList = new Map<string, string[]>();

	// Populate adjacency list with dependencies
	nodes.forEach(node => {
		if (node.parentId) {
			if (!adjacencyList.has(node.parentId)) {
				adjacencyList.set(node.parentId, []);
			}
			adjacencyList.get(node.parentId)!.push(node.id);
		}
	});

	// Array to hold the sorted result
	const sortedNodes: any[] = [];
	const visited = new Set<string>();

	// Helper function to perform DFS
	function dfs(nodeId: string) {
		if (visited.has(nodeId)) return;
		visited.add(nodeId);

		// Visit all children of the current node
		const children = adjacencyList.get(nodeId) || [];
		children.forEach(childId => dfs(childId));

		// Add the current node to the result after its children
		sortedNodes.push(nodeMap.get(nodeId));
	}

	// Start DFS from nodes with no parent or unvisited nodes
	nodes.forEach(node => {
		if (!node.parentId) {
			dfs(node.id);
		}
	});

	// Handle any unvisited nodes to account for disconnected parts of the graph
	nodes.forEach(node => {
		if (!visited.has(node.id)) {
			dfs(node.id);
		}
	});

	return sortedNodes.reverse();
}


const nodeWidth = 270;
const nodeHeight = 100;

export function getLayoutedElements(layerNodes: LayerNode[], layerEdges: LayerEdge[], direction: 'TB' | 'LR'): { nodes: Node[], edges: Edge[] } {
	const dagreGraph = new dagre.graphlib.Graph({
		directed: true,
		multigraph: true,
		compound: false
	});
	
	dagreGraph.setDefaultEdgeLabel(() => ({}));
	dagreGraph.setGraph({
		rankdir: direction,
		ranksep: 50,
		align: 'UL',
		nodesep: 50,
		marginx: 0,
		marginy: 0
	});

	layerNodes.forEach(node => {
		if (node.node_type !== 'container') {
			dagreGraph.setNode(node.id.toString(), {
				...node,
				width: nodeWidth,
				height: nodeHeight
			});
		}
		else {
			dagreGraph.setNode(node.id.toString(), {
				...node,
			});
		}
	});
	
	layerEdges.forEach(edge => {
		dagreGraph.setEdge(edge.source.toString(), edge.target.toString());
	});
	
	layerNodes.forEach(node => {
		if (node.node_type !== 'container') return;
		node.children.forEach(child => {
			console.log("Adding child", child, "to parent", node.id);
			dagreGraph.setParent(child.toString(), node.id.toString());
		});
	});
	
	dagre.layout(dagreGraph);
	
	const retNodes = dagreGraph.nodes().map(nodeId => layerNodes.find(node => node.id === parseInt(nodeId))!).map(node => ({
		id: node.id.toString(),
		position: {
			x: dagreGraph.node(node.id.toString()).x - dagreGraph.node(node.id.toString()).width / 2,
			y: dagreGraph.node(node.id.toString()).y - dagreGraph.node(node.id.toString()).height / 2
		},
		type: node.node_type,
		parentId: dagreGraph.parent(node.id.toString()),
		extent: "parent" as const,
		width: dagreGraph.node(node.id.toString()).width,
		height: dagreGraph.node(node.id.toString()).height,
		data: {
			id: node.id,
			name: node.name,
		}
	}))
	const retEdges = layerEdges.map(edge => ({
		source: edge.source.toString(),
		target: edge.target.toString(),
		id: `${edge.source}-${edge.target}`,
		animated: true,
		style: 'stroke: black',
		label: edge.label,
		data: { label: edge.label },
		type: 'defaultLayerEdge'
	}))
	
	const sortedNodes = sortNodesByParentDependency(retNodes);

	return { nodes: sortedNodes, edges: retEdges };
};

export function topologicalSort<T extends { id: string }>(nodes: T[], edges: { source: string, target: string }[]): T[] {
    const inDegree = new Map<string, number>();
    const adjacencyList = new Map<string, string[]>();

    // Initialize in-degree and adjacency list
    nodes.forEach(node => {
        inDegree.set(node.id, 0);
        adjacencyList.set(node.id, []);
    });

    // Build the graph
    edges.forEach(edge => {
        if (adjacencyList.has(edge.source)) {
            adjacencyList.get(edge.source)!.push(edge.target);
        }
        inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    });

    // Find all nodes with 0 in-degree
    const zeroInDegreeQueue: string[] = [];
    inDegree.forEach((degree, node) => {
        if (degree === 0) {
            zeroInDegreeQueue.push(node);
        }
    });

    const sortedNodes: T[] = [];
    while (zeroInDegreeQueue.length > 0) {
        const nodeId = zeroInDegreeQueue.shift()!;
        const node = nodes.find(n => n.id === nodeId)!;
        sortedNodes.push(node);

        // Decrease the in-degree of all adjacent nodes
        adjacencyList.get(nodeId)!.forEach(adjacentNodeId => {
            inDegree.set(adjacentNodeId, inDegree.get(adjacentNodeId)! - 1);
            if (inDegree.get(adjacentNodeId) === 0) {
                zeroInDegreeQueue.push(adjacentNodeId);
            }
        });
    }

    // Check for cycles in the graph
    if (sortedNodes.length !== nodes.length) {
        throw new Error("Graph has at least one cycle, topological sort not possible");
    }

    return sortedNodes;
}
