<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import type { NodeWithLayout } from '$lib/types';
	import { NodeColors } from '$lib/utils/utils';
	import * as Accordion from '$lib/components/ui/accordion';

	interface $$Props extends NodeProps {
		data: NodeWithLayout;
	}
	let { data }: $$Props = $props();
	
	console.log(data.layout_horizontal)

</script>

<Handle id={`${data.id}-target`} type="target" position={data.layout_horizontal ? Position.Left : Position.Top} />
<Handle id={`${data.id}-source`} type="source" position={data.layout_horizontal ? Position.Right : Position.Bottom} />

<div
	class="p-4 wrapper bg-white"
	class:tutorial_cnn_layer={data.tutorial_node}
	style:background-color={NodeColors[data.layer_type]}
>
	<h2 class="text-lg font-bold">{data.name}</h2>

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
</style>
