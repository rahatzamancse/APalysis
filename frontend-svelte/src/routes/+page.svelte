<script lang="ts">
	import { onMount } from 'svelte';
	import { getLayoutedElements } from '$lib/utils';
	import * as api from '$lib/api';
	import TensorNode from '$lib/components/TensorNode.svelte';
	import ContainerNode from '$lib/components/ContainerNode.svelte';
	import FunctionNode from '$lib/components/FunctionNode.svelte';
	import { refreshData } from '$lib/stores';
	import CustomEdge from '$lib/components/CustomEdge.svelte';

	// import { useTour } from '@reactour/tour';
	import {
		SvelteFlow,
		Controls,
		Background,
		Position,
		MiniMap,
		type Node,
		type Edge,
		ControlButton,
		getNodesBounds,
		getViewportForBounds,
		useSvelteFlow,
		SvelteFlowProvider
	} from '@xyflow/svelte';
	import '@xyflow/svelte/dist/style.css';

	import { toPng } from 'html-to-image';
	import LayerNode from '$lib/components/FunctionNode.svelte';

	import { writable } from 'svelte/store';
	
	const nodeTypes = {
		function: FunctionNode,
		tensor: TensorNode,
		container: ContainerNode
	};
	const edgeTypes = {
		defaultLayerEdge: CustomEdge
	};

	
	let flowRef: SvelteFlow = $state();

	let nodes = writable<Node[]>([{
		id: '0',
		data: { label: 'Loading the model...' },
		position: { x: 0, y: 0 }
	}]);
	let edges = writable<Edge[]>([]);
	
	async function fetchUpdatedData() {
		const updatedGraph = await api.getModelGraph();
		const layoutGraph = getLayoutedElements(updatedGraph.nodes, updatedGraph.edges, 'LR');
		layoutGraph.nodes.forEach(node => {
			console.log(node.id, node.type, node.parentId);
		});
		// layoutGraph.edges.forEach(edge => {
		// 	const sourceNode = updatedGraph.nodes.find(node => node.id.toString() === edge.source)!;
		// 	const targetNode = updatedGraph.nodes.find(node => node.id.toString() === edge.target)!;
		// 	const sourceType = sourceNode.node_type;
		// 	const targetType = targetNode.node_type;
		// 	if (sourceType === 'container' || targetType !== 'container') {
		// 		console.log(`${sourceNode.name} (${sourceType}) -> ${targetNode.name} (${targetType})`)
		// 	}
		// });
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
	
	function handleExportFlow() {
		// const flow = toObject();
		// const json = JSON.stringify(flow, null, 2);
		// const blob = new Blob([json], { type: 'application/json' });
		// const url = URL.createObjectURL(blob);
		// const a = document.createElement('a');
		// a.href = url;
		// a.download = 'node-placement.json';
		// a.click();
		// URL.revokeObjectURL(url);
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
		bind:this={flowRef}
	>
		<MiniMap pannable zoomable style="border: 1px solid lightgray;" />
		<Controls>
			<!-- <ControlButton
				onclick={handleLayoutChange}
				title="Change layout between vertical and horizontal."
			>
				{isHorizontal ? 'H' : 'V'}
			</ControlButton> -->
			<ControlButton onclick={handleScreenshot}>
				<img src="/ControlButtons/save.png" alt="Export" width="16px" height="16px" />
			</ControlButton>
			<ControlButton
				onclick={() => {
					const input = document.createElement('input');
					input.type = 'file';
					input.onchange = async (e) => {
						const file = (e.target as HTMLInputElement).files?.[0];
						if (file) {
							const reader = new FileReader();
							reader.onload = async (e) => {
								if (e.target) {
									const text = (e.target as any).result;
									if (text) {
										const flow = JSON.parse(text);
										if (flow) {
											const { x = 0, y = 0, zoom = 1 } = flow.viewport;
											$nodes = flow.nodes || [];
											$edges = flow.edges || [];
										}
									}
								}
							};
							reader.readAsText(file);
						}
					};
					input.click();
				}}
			>
				<img src="/ControlButtons/import.png" alt="Load" width="16px" height="16px" />
			</ControlButton>
			<ControlButton
				onclick={handleExportFlow}
			>
				<img src="ControlButtons/export.png" alt="Save" width="16px" height="16px" />
			</ControlButton>
		</Controls>
		<Background 
			bgColor="#dee5ed"
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
	
	:global(.svelte-flow__edge-label) {
		z-index: 200000;
		background-color: red;
	}
</style>
