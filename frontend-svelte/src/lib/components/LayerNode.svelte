<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import type { NodeWithLayout, Node, Edge } from '$lib/types';
	import { NodeColors } from '$lib/utils/utils';
	import * as Accordion from '$lib/components/ui/accordion';
	import * as api from '$lib/api';
	import { createGraph } from '$lib/graph.svelte';

	interface $$Props extends NodeProps {
		data: NodeWithLayout;
	}
	let { data }: $$Props = $props();
	let graph = createGraph();
	
	function handleExpand() {
		api.expandNode(data.id).then((data) => {
			graph.setNodes(data.nodes);
			graph.setEdges(data.edges);
		});
	}
	
</script>

<Handle id={`${data.id}-target`} type="target" position={data.layout_horizontal ? Position.Left : Position.Top} />
<Handle id={`${data.id}-source`} type="source" position={data.layout_horizontal ? Position.Right : Position.Bottom} />

<div
	class="p-4 wrapper bg-white"
	class:tutorial_cnn_layer={data.tutorial_node}
	style:background-color={NodeColors[data.layer_type]}
>
	<h2 class="text-lg font-bold">{data.name}</h2>
	
	{#if !data.is_leaf}
		<button class="expand-button" onclick={handleExpand}></button>
	{/if}

	<Accordion.Root class="w-full" multiple>
		<Accordion.Item value="details">
			<Accordion.Trigger>Details</Accordion.Trigger>
			<Accordion.Content>
				<ul>
					{#if data.layer_type}
						<li><b>Layer :</b> {data.layer_type}</li>
					{/if}
					{#if data.input_shape}
						<li><b>Input :</b> ({data.input_shape.toString()})</li>
					{/if}
					{#if data.kernel_size}
						<li><b>Kernel Shape :</b> ({data.kernel_size.toString()})</li>
					{/if}
					{#if data.out_edge_weight}
						<li><b>Kernel # :</b> {data.out_edge_weight.length}</li>
					{/if}
					{#if data.output_shape}
						<li><b>Output :</b> ({data.output_shape.toString()})</li>
					{/if}
				</ul>
			</Accordion.Content>
		</Accordion.Item>
	</Accordion.Root>
</div>

<style>
	.wrapper {
		border-radius: 20px;
		border: 1px solid #aaa;
	}
	
	.expand-button {
		width: 10px;
		height: 10px;
		background-color: #aaa;
		border-radius: 50%;
		position: absolute;
		top: 20px;
		right: 20px;
	}
</style>