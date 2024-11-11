<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import type { TensorNode } from '$lib/types';

	type $$Props = NodeProps & {
		data: TensorNode;
	};
	let { data }: $$Props = $props();
	
	function getNodeName(nodeName: string, nodeId: number) {
		let name = nodeName?.split("->").slice(-1)[0] || nodeId.toString().split("->").slice(-1)[0].split('=')[1];
		if (name === 'hidden-tensor') {
			name = 'Tensor';
		}
		return name;
	}
	const name = getNodeName(data.name, data.id);
</script>

<Handle id={`${data.id}-target`} type="target" position={Position.Left} />
<Handle id={`${data.id}-source`} type="source" position={Position.Right} />

<div class="wrapper">
	<div class="header">
		<h2 class="text-lg font-bold p-1 pl-5">
			{name}
		</h2>
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