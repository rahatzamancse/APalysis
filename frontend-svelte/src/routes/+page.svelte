<script lang="ts">
	import * as api from '$lib/api';
	import CustomEdge from '$lib/components/CustomEdge.svelte';
	import ContainerNode from '$lib/components/LayerNodes/ContainerNode.svelte';
	import FunctionNode from '$lib/components/LayerNodes/FunctionNode.svelte';
	import TensorNode from '$lib/components/LayerNodes/TensorNode.svelte';
	import TensorWindow from '$lib/components/WindowNodes/TensorWindow.svelte';
	// import FunctionWindow from '$lib/components/WindowNodes/FunctionWindow.svelte';
	import { refreshData } from '$lib/stores';
	import { getLayoutedElements } from '$lib/utils/utils';
	import { onMount } from 'svelte';
	import {
		Background,
		ControlButton,
		Controls,
		getNodesBounds,
		getViewportForBounds,
		MiniMap,
		SvelteFlow,
		SvelteFlowProvider,
		type Edge,
		type EdgeTypes,
		type Node,
		type NodeTypes,
	} from '@xyflow/svelte';
	import '@xyflow/svelte/dist/style.css';
	import type { TensorWindowData, FunctionWindowData } from '$lib/types';

	import { toPng } from 'html-to-image';
	
	import { writable } from 'svelte/store';
	
	const nodeTypes: NodeTypes = {
		function: FunctionNode,
		tensor: TensorNode,
		container: ContainerNode,
		tensorWindow: TensorWindow,
		// functionWindow: FunctionWindow
	};
	const edgeTypes: EdgeTypes = {
		smoothstep: CustomEdge
	};

	let nodes = writable<Node[]>([{
		id: '0',
		data: { label: 'Loading the model...' },
		position: { x: 0, y: 0 }
	}]);
	let edges = writable<Edge[]>([]);
	
	async function fetchUpdatedData() {
		const updatedGraph = await api.getModelGraph();
		const layoutGraph = await getLayoutedElements(updatedGraph.nodes, updatedGraph.edges, 'LR');
		layoutGraph.nodes.forEach(node => {
			if (['tensor', 'function'].includes(node.data.node_type as string)) {
				node.data.addWindow = addWindowNode;
			}
		});
		$nodes = layoutGraph.nodes;
		$edges = layoutGraph.edges;
	}

	onMount(fetchUpdatedData);
	
	function handleScreenshot() {
		const imageWidth = 1000;
		const imageHeight = 1000;
		const nodesBounds = getNodesBounds($nodes);
		const viewport = getViewportForBounds(nodesBounds, imageWidth, imageHeight, 0.5, 2.0, 0.2);

		const viewportDomNode = document.querySelector<HTMLElement>('.svelte-flow__viewport')!;

		if (viewport) {
			toPng(viewportDomNode, {
				backgroundColor: '#1a365d',
				width: imageWidth,
				height: imageHeight,
				style: {
					width: `${imageWidth}px`,
					height: `${imageHeight}px`,
					transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`
				}
			}).then((dataUrl) => {
				const link = document.createElement('a');
				link.download = 'svelte-flow.png';
				link.href = dataUrl;
				link.click();
			});
		}
	}
	
	function addWindowNode(from: string, newNodeData: TensorWindowData | FunctionWindowData, width: number = 500, height: number = 500) {
		switch(newNodeData.type) {
			case 'tensorWindow':
				return addTensorWindowNode(from, newNodeData, width, height);
			case 'functionWindow':
				// TODO: Implement function window handling
				console.warn('Function window not yet implemented');
				return;
		}
	}

	function addTensorWindowNode(from: string, newNodeData: TensorWindowData, width: number, height: number) {
		const NODE_OFFSET_X = 0;
		const NODE_OFFSET_Y = 10;
		
		const fromNode = $nodes.find(node => node.id.toString() === from.toString());
		if (!fromNode) {
			console.error(`Node with id ${from} not found while adding TensorWindowNode`);
			return;
		}
		
		const newNode: Node = {
			id: newNodeData.id.toString(),
			data: newNodeData as unknown as Record<string, unknown>,
			position: {
				x: fromNode.position.x + (fromNode.width ?? 10) + NODE_OFFSET_X,
				y: fromNode.position.y + (fromNode.height ?? 10) + NODE_OFFSET_Y
			},
			width: width,
			height: height,
			parentId: fromNode.parentId,
			type: 'tensorWindow',
			extent: 'parent' as const,
		}
		
		const newEdge: Edge = {
			id: `${from}-${newNode.id}-window`,
			source: from,
			// sourceHandle: `${from}-window-output`,
			target: newNode.id,
			// targetHandle: `${newNode.id}-source`,
			type: 'smoothstep',
		}
		
		// Recursively update parent heights if needed
		let currentParentId = fromNode.parentId;
		while (currentParentId) {
			const parentNode = $nodes.find(node => node.id === currentParentId);
			if (!parentNode) break;
			
			// Calculate required height based on from node and new node positions
			const requiredHeight = (fromNode.height || 0) + (newNode.height || 0);
			const requiredWidth = (fromNode.width || 0) + (newNode.width || 0);
			
			// Update parent height if required height is larger
			if (parentNode.height && requiredHeight > parentNode.height) {
				parentNode.height = requiredHeight + 100; // Add padding
				$nodes = $nodes.map(node => 
					node.id === parentNode.id ? parentNode : node
				);
			}
			if (parentNode.width && requiredWidth > parentNode.width) {
				parentNode.width = requiredWidth + 100; // Add padding
				$nodes = $nodes.map(node => 
					node.id === parentNode.id ? parentNode : node
				);
			}
			
			// Move up to next parent
			currentParentId = parentNode.parentId;
		}
		
		$edges = [...$edges, newEdge];
		$nodes = [...$nodes, newNode];
	}
	
	// Listen to the refreshData store
	refreshData.subscribe((value) => {
		if (value) {
			fetchUpdatedData().then(() => {
				refreshData.set(false); // Reset the store
			});
		}
	});
</script>

<svelte:head>
	<title>Home</title>
	<meta name="description" content="Svelte demo app" />
</svelte:head>

<div class="flow-container" >
	<SvelteFlowProvider>
		
	<SvelteFlow
		{nodes}
		{edges}
		{nodeTypes}
		{edgeTypes}
		colorMode="light"
		fitViewOptions={{ padding: 0.1, duration: 1000 }}
		attributionPosition="bottom-right"
		minZoom={0.1}
		maxZoom={10}
	>
		<MiniMap pannable zoomable style="border: 1px solid lightgray;" />
		<Controls>
			<ControlButton onclick={handleScreenshot}>
				<img src="/ControlButtons/save.png" alt="Export" width="16px" height="16px" />
			</ControlButton>
		</Controls>
		<Background 
			bgColor="#e6e6e6"
			patternColor="#aaa"
		/>
	</SvelteFlow>
	</SvelteFlowProvider>
</div>


<style>
	.flow-container {
		height: 100vh;
		width: 100vw;
		position: absolute;
		top: 0;
		left: 0;
	}
	:global(.svelte-flow__edges) {
		z-index: 100;
	}
</style>
