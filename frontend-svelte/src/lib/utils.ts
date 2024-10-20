import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { cubicOut } from "svelte/easing";
import type { TransitionConfig } from "svelte/transition";
import dagre from '@dagrejs/dagre';
import type { LayerNode, LayerEdge } from '$lib/types';
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


const nodeWidth = 200;
const nodeHeight = 100;

export function getLayoutedElements(layerNodes: LayerNode[], layerEdges: LayerEdge[], direction: 'TB' | 'LR'): { nodes: Node[], edges: Edge[] } {
	const dagreGraph = new dagre.graphlib.Graph({
		directed: true,
		multigraph: true,
		compound: true
	});
	
	dagreGraph.setDefaultEdgeLabel(() => ({}));
	dagreGraph.setGraph({
		rankdir: direction,
		ranksep: 40,
		align: 'UL',
		nodesep: 120,
		marginx: 0,
		marginy: 0
	});

	layerNodes.forEach(node => {
		if (node.is_leaf || !node.expanded) {
			dagreGraph.setNode(node.id, {
				...node,
				width: nodeWidth,
				height: nodeHeight
			});
		}
		else {
			dagreGraph.setNode(node.id, node)
		}
	});
	
	layerEdges
		.filter(edge => edge.edge_type === 'data_flow')
		// dagre does not support edges for compound nodes
		.filter(edge => !layerEdges.some(e => e.source === edge.target && e.target === edge.source && e.edge_type === 'parent'))
		.filter(edge => {
			const sourceNode = layerNodes.find(node => node.id === edge.source);
			const targetNode = layerNodes.find(node => node.id === edge.target);
			return !((sourceNode && sourceNode.expanded && !sourceNode.is_leaf) || (targetNode && targetNode.expanded && !targetNode.is_leaf));
		})
		.forEach(edge => {
			dagreGraph.setEdge(edge.source, edge.target);
		});
		
	layerEdges
		.filter(edge => edge.edge_type === 'parent')
		.forEach(edge => {
			console.log("Setting parent", edge.target, edge.source);
			dagreGraph.setParent(edge.target, edge.source);
		});
	
	dagre.layout(dagreGraph);
	
	const sortedNodes = topologicalSort(dagreGraph.nodes().map(n => ({ id: n })), layerEdges.filter(edge => edge.edge_type === 'parent').map(edge => ({
		source: edge.source,
		target: edge.target
	})));
	
	console.log("Dagre Graph", dagreGraph);
	layerNodes.forEach(node => {
		console.log(node.id, dagreGraph.node(node.id).x, dagreGraph.node(node.id).y, dagreGraph.node(node.id).width, dagreGraph.node(node.id).height);
	});
	
	return {
		nodes: sortedNodes.map(node => layerNodes.find(ln => ln.id === node.id)!).map(node => ({
			id: node.id,
			position: {
				x: dagreGraph.node(node.id).x - dagreGraph.node(node.id).width / 2,
				y: dagreGraph.node(node.id).y - dagreGraph.node(node.id).height / 2
			},
			type: node.is_leaf ? 'layerNode' : node.expanded ? 'parentExpanded' : 'parentCollapsed',
			parentId: layerEdges.find(edge => edge.target === node.id && edge.edge_type === 'parent')?.source,
			extent: "parent" as const,
			width: dagreGraph.node(node.id).width,
			height: dagreGraph.node(node.id).height,
			data: {
				id: node.id,
				name: node.name,
				layer_type: node.layer_type,
				tensor_type: node.tensor_type,
				output_shape: node.output_shape,
				is_leaf: node.is_leaf,
				expanded: node.expanded,
			}
		})),
		edges: layerEdges.filter(edge => edge.edge_type === 'data_flow').map(edge => ({
			source: edge.source,
			target: edge.target,
			id: `${edge.source}-${edge.target}`,
			animated: true,
			style: 'stroke: black',
			label: '(' + JSON.stringify(layerNodes.find(node => node.id === edge.source)?.output_shape) + ')',
			data: { edge_type: edge.edge_type },
			zIndex: 100,
			type: 'defaultLayerEdge'
		}))
	};
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
