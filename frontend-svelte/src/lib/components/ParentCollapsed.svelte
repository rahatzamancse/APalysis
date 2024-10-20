<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { NodeColors } from '$lib/utils/utils';
	import { refreshData } from '$lib/stores';
	import * as api from '$lib/api';

	type $$Props = NodeProps & {
		data: {
			id: string;
			name?: string;
		};
	};
	let { data }: $$Props = $props();
	
	let name = data.name?.split("->").slice(-1)[0] || data.id.split("->").slice(-1)[0];
	name = name.split('=')[1];

	function handleExpand() {
		api.expandNode(data.id).then((modelGraph) => {
			refreshData.set(true); // Signal to refresh data
		});
	}
</script>

<Handle id={`${data.id}-target`} type="target" position={Position.Left} />
<Handle id={`${data.id}-source`} type="source" position={Position.Right} />

<div class="wrapper">
	<div class="header">
		<h2 class="text-lg font-bold p-1 pl-5">
			{name}
		</h2>
		<button class="expand-button" onclick={handleExpand}>
			<img src="/expand_logo.png" alt="Expand" width=100% height=100% />
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
		background-color: #f29a28;
		width: 100%;
		position: relative;
	}
	.expand-button {
		width: 20px;
		height: 20px;
		position: absolute;
		top: 50%;
		right: 10px;
		transform: translateY(-50%);
		transition: background-color 0.3s ease;
		transition: width 0.3s ease, height 0.3s ease;
	}
	.expand-button:hover {
		background-color: lightgray;
		width: 25px;
		height: 25px;
	}
</style>
