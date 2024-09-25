<script lang="ts">
	import { onMount } from 'svelte';
	import * as api from '$lib/api';
	import dagre from '@dagrejs/dagre';
	import { createGraph } from '$lib/graph.svelte';

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
	import LayerNode from '$lib/components/LayerNode.svelte';

	import { writable } from 'svelte/store';
	
	const nodeTypes = { layerNode: LayerNode };

	const dagreGraph = new dagre.graphlib.Graph();
	dagreGraph.setDefaultEdgeLabel(() => ({}));

	const nodeWidth = 200;
	const nodeHeight = 100;
	let isHorizontal = true;
	let flowRef: SvelteFlow;

	const getLayoutedElements = (nodes: Node[], edges: Edge[], direction: 'TB' | 'LR' = isHorizontal ? 'LR' : 'TB') => {
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

	let nodes = writable<Node[]>(initialNodes);
	let edges = writable<Edge[]>(initialEdges);

	function onLayoutChange(isHorizontal: boolean) {
		const direction = isHorizontal ? 'LR' : 'TB';
		const layoutedElements = getLayoutedElements($nodes, $edges, direction);

		$nodes = layoutedElements.nodes;
		
		// We need to mutate the nodes' data to update the layout_horizontal property
		$nodes.forEach((node, i) => {
			node.data = {
				...node.data,
				layout_horizontal: isHorizontal
			};
		});

		$nodes = $nodes

		$edges = [
			...layoutedElements.edges
		]
	}

	onMount(() => {
		api.getModelGraph().then((modelGraph) => {
			const newNodes: Node[] = modelGraph.nodes.map((node) => ({
				id: node.id,
				position: { x: 0, y: 0 },
				data: {
					id: node.id,
					label: node.label,
					layer_type: node.layer_type,
					name: node.name,
					tensor_type: node.tensor_type,
					output_shape: node.output_shape,
					layout_horizontal: isHorizontal,
					tutorial_node: false,
					position: { x: 0, y: 0 },
					is_parent: node.is_parent,
					parent: node.parent,
				},
				type: 'layerNode',
				// type: node.is_parent ? 'group' : 'layerNode',
				// parentId: node.parent ? '1=' + node.parent : null,
				// style: node.is_parent ? 'stroke: white; stroke-width: 2px; width: 2000px; height: 2000px;' : ''
			}));
			
			const newEdges: Edge[] = modelGraph.edges.map((edge) => ({
				source: edge.source,
				target: edge.target,

				id: `${edge.source}-${edge.target}`,
				animated: true,
				style: 'stroke: white',
				label: '(' + JSON.stringify(newNodes.find((node) => node.id === edge.source)?.data.output_shape) + ')'
			}))
			
			console.log("New Nodes", newNodes)
			console.log("New Edges", newEdges)

			const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
				newNodes,
				newEdges,
				isHorizontal ? 'LR' : 'TB'
			);
			$nodes = layoutedNodes;
			$edges = layoutedEdges;
		});
	});

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
	
	function handleLayoutChange() {
		isHorizontal = !isHorizontal;
		onLayoutChange(isHorizontal);
	}
</script>

<svelte:head>
	<title>Home</title>
	<meta name="description" content="Svelte demo app" />
</svelte:head>

<div
	style:height="100vh"
	style:width="100vw"
	style:position="absolute"
	style:top="0"
	style:left="0"
	style:border="1px solid #000"
>
	<SvelteFlowProvider>
		
	<SvelteFlow
		{nodes}
		{edges}
		{nodeTypes}
		colorMode="dark"
		fitViewOptions={{ padding: 0.1, duration: 1000 }}
		attributionPosition="bottom-right"
		minZoom={0.1}
		maxZoom={10}
		bind:this={flowRef}
	>
		<MiniMap pannable zoomable style="border: 1px solid #000;" />
		<Controls>
			<ControlButton
				onclick={handleLayoutChange}
				title="Change layout between vertical and horizontal."
			>
				{isHorizontal ? 'H' : 'V'}
			</ControlButton>
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
		<Background />
	</SvelteFlow>
	</SvelteFlowProvider>
</div>

<style>
</style>
