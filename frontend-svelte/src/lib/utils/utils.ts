import type { LayerEdge, LayerNode } from '$lib/types';
import type { Edge, Node } from '@xyflow/svelte';
import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";

export const NodeColors: { [key: string]: string } = {
    'Conv2D': '#CBE4F9',
    'Dense': '#CDF5F6',
    'InputLayer': '#EFF9DA',
    'MaxPooling2D': '#FFFF00',
    'Flatten': '#F9EBDF',
    'Dropout': '#FF00FF',
    'Activation': '#FF8000',
    'GlobalAveragePooling2D': '#8000FF',
    'GlobalMaxPooling2D': '#0080FF',
    'BatchNormalization': '#D6CDEA',
    'Add': '#80FF00',
    'Concatenate': '#F9D8D6',
    'AveragePooling2D': '#800080',
    'ZeroPadding2D': '#808000',
    'UpSampling2D': '#008000',
    'Reshape': '#800000',
    'Permute': '#000080',
    'RepeatVector': '#808080',
    'Lambda': '#008080',
}

export function chunkify<T>(arr: T[], size: number): T[][] {
    return [...Array(Math.ceil(arr.length / size))].map((_, i) =>
        arr.slice(size * i, size + size * i)
    );
}

export function transposeArray<T>(array: T[][]): T[][] {
    if (array === undefined || array.length === 0) return [];
    return array[0].map((_, j) =>
        array.map((row) => row[j])
    );
}
export function findIndicesOfMax(inp: number[], count: number) {
    const outp = [];
    for (let i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function(a, b) { return inp[b] - inp[a]; }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}
export function calcAllPairwiseDistance(arr: number[]) {
    let sum = 0
    for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            sum += Math.abs(arr[i] - arr[j])
        }
    }
    return sum
}

export function calcVariance(inp: number[]) {
    const mean = inp.reduce((a, b) => a + b, 0) / inp.length
    const variance = inp.map(item => Math.pow(item - mean, 2)).reduce((a, b) => a + b, 0) / inp.length
    return variance
}
export function calcPairwiseDistance(arr1: number[], arr2: number[]) {
    let sum = 0
    for (let i = 0; i < arr1.length; i++) {
        sum += Math.pow(arr1[i] - arr2[i], 2)
    }
    return Math.sqrt(sum)
}

export function calcSumPairwiseDistance(...arrs: number[][]): number {
    let sum = 0
    for (let i = 0; i < arrs.length; i++) {
        for (let j = i + 1; j < arrs.length; j++) {
            sum += calcPairwiseDistance(arrs[i], arrs[j])
        }
    }
    return sum
}

export function getRawHeatmap(heatmap: number[][], nExamples: number) {
    return heatmap.slice(0, nExamples)
}

export function shortenName(name: string, len: number): string {
    name = name.split(": ")[0]
    return name.length<=len ? name : name.slice(0, len) + '...'
}

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

function buildElkNode(node: LayerNode, allNodes: LayerNode[]): ElkNode {
	const baseElkNode: ElkNode = {
		id: node.id.toString(),
	};
	if(node.node_type !== 'container') {
		baseElkNode.width = nodeWidth;
		baseElkNode.height = nodeHeight;
	}

	// If the node is a ContainerNode, recursively add its children
	if (node.node_type === 'container') {
		baseElkNode.children = node.children.map(childId => {
			const childNode = allNodes.find(n => n.id === childId);
			return childNode ? buildElkNode(childNode, allNodes) : null;
		}).filter((n): n is ElkNode => n !== null); // Filter out null if any children are not found
	}

	return baseElkNode;
}


function updateNodePositions(layoutedNode: ElkNode, originalNodes: Node[]) {
	const originalNode = originalNodes.find(n => n.id === layoutedNode.id)!;
	if (layoutedNode.x !== undefined && layoutedNode.y !== undefined) {
		originalNode.position.x = layoutedNode.x;
		originalNode.position.y = layoutedNode.y;
	}
	if (layoutedNode.height !== undefined) {
		originalNode.height = layoutedNode.height;
	}
	if (layoutedNode.width !== undefined) {
		originalNode.width = layoutedNode.width;
	}

	// Recursively update positions for child nodes if present
	layoutedNode.children?.forEach(childLayoutedNode => {
		updateNodePositions(childLayoutedNode, originalNodes);
	});
}

const nodeWidth = 160;
const nodeHeight = 160;

export async function getLayoutedElements(layerNodes: LayerNode[], layerEdges: LayerEdge[], direction: 'TB' | 'LR'): Promise<{ nodes: Node[], edges: Edge[] }> {
	const elk = new ELK();
	// Identify top-level nodes (not a child of any container)
	const topLevelNodes = layerNodes.filter(node =>
		!layerNodes.some(parent => parent.node_type === 'container' && parent.children.includes(node.id))
	);

	// Build the ELK graph with top-level nodes and edges
	const elkGraph: ElkNode = {
		id: 'root',
		layoutOptions: {
			"elk.algorithm": "layered",
			"elk.alignment": "AUTOMATIC",
			"elk.layered.spacing.nodeNodeBetweenLayers": "100",
			"elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
			"elk.direction": direction,
			"elk.hierarchyHandling": "INCLUDE_CHILDREN",
			"elk.spacing.nodeNode": "60",
		},
		children: topLevelNodes.map(node => buildElkNode(node, layerNodes)).map(n => ({
			...n,
			layoutOptions: {
				"elk.padding": "[top=50,left=20,bottom=50,right=20]"
			}
		})),
		// @ts-ignore You really don't need the sources and targets, source and target is enough
		edges: layerEdges.map(edge => ({
			id: `${edge.source}-${edge.target}`,
			source: edge.source.toString(),
			target: edge.target.toString(),
			sourceHandle: `${edge.source}-output`,
			targetHandle: `${edge.target}-input`,
			type: 'smoothstep',
			animated: true
		}))
	};

	const layoutedElkGraph = await elk.layout(elkGraph);
	
	const retNodes: Node[] = layerNodes.map(node => ({
		id: node.id.toString(),
		position: {
			x: 0,
			y: 0
		},
		type: node.node_type,
		parentId: layerNodes.filter(n => n.node_type === 'container').find(n => n.children?.includes(node.id))?.id.toString(),
		extent: "parent" as const,
		data: {
			...node,
			id: node.id,
			name: node.name,
		}
	}))
	
	layoutedElkGraph.children?.forEach(layoutedElkNode => {
		updateNodePositions(layoutedElkNode, retNodes);
	})
	
	// @ts-ignore Same as above
	const retEdges: Edge[] = layoutedElkGraph.edges!
	
	const sortedNodes = sortNodesByParentDependency(retNodes);

	return { nodes: sortedNodes, edges: retEdges };
}

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