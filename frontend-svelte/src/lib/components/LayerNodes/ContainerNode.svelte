<script lang="ts">
	import { Handle, Position, type NodeProps, NodeResizeControl } from '@xyflow/svelte';
	import type { ContainerNode } from '$lib/types';

	type $$Props = NodeProps & {
		data: ContainerNode;
	};
	const { data }: $$Props = $props();
	const name =
		data.name?.split('->').slice(-1)[0] ||
		data.id.toString().split('->').slice(-1)[0].split('=')[1];
</script>

<Handle id={`${data.id}-input`} type="target" position={Position.Left} />
<Handle id={`${data.id}-output`} type="source" position={Position.Right} />

<div class="wrapper">
	<h2 class="text-lg font-bold p-1 pl-5">
		{name}
	</h2>
	<NodeResizeControl class="resize-icon" minWidth={100} minHeight={50}>
		<svg
			xmlns="http://www.w3.org/2000/svg"
			width="20"
			height="20"
			viewBox="0 0 24 24"
			stroke-width=2
			stroke="#ff0071"
			fill="none"
			stroke-linecap="round"
			stroke-linejoin="round"
			style="position: absolute; right: 5; bottom: 5"
		>
			<path stroke="none" d="M0 0h24v24H0z" fill="none" />
			<polyline points="16 20 20 20 20 16" />
			<line x1="14" y1="14" x2="20" y2="20" />
			<polyline points="8 4 4 4 4 8" />
			<line x1="4" y1="4" x2="10" y2="10" />
		</svg>
	</NodeResizeControl>
</div>

<style>
	.wrapper {
		border-radius: 14px;
		border: 1px solid #aaa;
		background-color: #f0f0f0;
		min-height: 100%;
		overflow: hidden;
	}
	:global(.resize-icon) {
		background-color: transparent;
		border: none;
	}
</style>
