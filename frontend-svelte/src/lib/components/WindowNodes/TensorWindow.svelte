<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import ActivationProjection from '$lib/components/Visualizations/ActivationProjection.svelte';
	import type { TensorWindowData } from '$lib/types';
	
	type $$Props = NodeProps & {
		data: TensorWindowData;
	};
	let { data }: $$Props = $props();
	
	function getNodeName(nodeName: string, nodeId: string) {
		let name = nodeName?.split("->").slice(-1)[0] || nodeId.split("->").slice(-1)[0].split('=')[1];
		if (name === 'hidden-tensor') {
			name = 'Tensor';
		}
		return name;
	}
	const name = getNodeName(data.name, data.id);
	
	
</script>

<Handle id={`${data.id}-source`} type="target" position={Position.Top} />
<div class="wrapper">
	<div class="header">
		<h2 class="text-lg font-bold p-1 pl-5">
			{name}
		</h2>
	</div>
	<div class="content">
		<ActivationProjection tensorId={data.tensorId} shape={data.shape} />
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