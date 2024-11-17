<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import type { TensorNode } from '$lib/types';
	import Accordion from '$lib/components/Accordion.svelte';

	type $$Props = NodeProps & {
		data: TensorNode;
	};
	let { data }: $$Props = $props();
	
	let windowIds = $state<string[]>([]);
	
	function getNodeName(nodeName: string, nodeId: string) {
		let name = nodeName?.split("->").slice(-1)[0] || nodeId.split("->").slice(-1)[0].split('=')[1];
		if (name === 'hidden-tensor') {
			name = 'Tensor';
		}
		return name;
	}
	const name = getNodeName(data.name, data.id);
	
	const addWindow = data.addWindow;
</script>

<Handle id={`${data.id}-input`} type="target" position={Position.Left} />
<Handle id={`${data.id}-output`} type="source" position={Position.Right} />
<Handle id={`${data.id}-window-output`} type="source" position={Position.Bottom} />

<div class="wrapper">
	<div class="header">
		<h2 class="text-lg font-bold p-1 pl-5">
			{name}
		</h2>
	</div>
	<div class="content">
		<Accordion>
			{#snippet header()}
				<h3>Details</h3>
			{/snippet}
			{#snippet details()}
				<ul>
					<li><b>Shape:</b> {data.shape}</li>
				</ul>
			{/snippet}
		</Accordion>
		
		<button class="show-value" onclick={() => {
			const newId = Math.random().toString(36).substring(2, 15);
			windowIds = [...windowIds, newId];
			addWindow && addWindow(
				data.id,
				{
					id: newId,
					fromId: data.id,
					name: name,
					type: 'tensorWindow',
					tensorId: data.id,
					shape: data.shape
				}
			)
		}}>
			Scatter Plot
		</button>
	</div>
</div>

<style>
	.wrapper {
		border-radius: 14px;
		border: 1px solid #aaa;
		background-color: #f0f0f0;
		min-height: 100%;
		overflow: hidden;
	}
	.header {
		background-color: lightgreen;
		width: 100%;
		position: relative;
	}
</style>