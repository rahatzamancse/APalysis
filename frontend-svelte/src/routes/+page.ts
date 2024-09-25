import { createGraph } from '$lib/graph.svelte';
import type { Node, Edge } from '@xyflow/svelte';
import dagre from '@dagrejs/dagre';
import * as api from '$lib/api';

const nodeWidth = 200;
const nodeHeight = 100;

const getLayoutedElements = (nodes: Node[], edges: Edge[], direction: 'TB' | 'LR') => {
	const dagreGraph = new dagre.graphlib.Graph();
	dagreGraph.setDefaultEdgeLabel(() => ({}));

	dagreGraph.setGraph({
		rankdir: direction,
		nodesep: 50,
		ranksep: 50,
		marginx: 50,
		marginy: 50
	});

	nodes.forEach((node) => {
		dagreGraph.setNode(node.id, {
			width: nodeWidth,
			height: nodeHeight
		});
	});

	edges.forEach((edge) => {
		dagreGraph.setEdge(edge.source, edge.target);
	});

	dagre.layout(dagreGraph);

	nodes.forEach((node) => {
		const nodeWithPosition = dagreGraph.node(node.id);
		node.position = {
			x: nodeWithPosition.x - nodeWidth / 2,
			y: nodeWithPosition.y - nodeHeight / 2
		};
		node.data.position = node.position;
	});

	return { nodes, edges };
};

export function load() {
	const graph = createGraph();
	
	let nodes = $derived(graph.nodes);
	let edges = $derived(graph.edges);

	const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
		nodes,
		edges,
		'LR'
	);
	
	nodes = layoutedNodes;
	edges = layoutedEdges;
	
	return graph;
}
