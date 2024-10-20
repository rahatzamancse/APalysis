<script lang="ts">
	import {
		getBezierPath,
		BaseEdge,
		type EdgeProps,
		EdgeLabelRenderer,
		useEdges
	} from '@xyflow/svelte';

	type $$Props = EdgeProps;
	
	export let id: $$Props['id'];
	export let sourceX: $$Props['sourceX'];
	export let sourceY: $$Props['sourceY'];
	export let sourcePosition: $$Props['sourcePosition'];
	export let targetX: $$Props['targetX'];
	export let targetY: $$Props['targetY'];
	export let targetPosition: $$Props['targetPosition'];
	export let markerEnd: $$Props['markerEnd'] = undefined;
	export let style: $$Props['style'] = undefined;
	export let label: $$Props['label'] = undefined;
	
	$: [edgePath, labelX, labelY] = getBezierPath({
		sourceX,
		sourceY,
		sourcePosition,
		targetX,
		targetY,
		targetPosition
	});

	const edges = useEdges();

	const onEdgeClick = () => {
		// Remove the edge
		// edges.update((eds) => eds.filter((edge) => edge.id !== id));
	};
</script>

<BaseEdge path={edgePath} {markerEnd} {style} />
<EdgeLabelRenderer>
	<!-- <div class="edge-label" role="button" tabindex="0" on:click={onEdgeClick} on:keydown={(e) => e.key === 'Enter' && onEdgeClick()}>
		{label}
	</div> -->
</EdgeLabelRenderer>

<style>
	/* .edge-label {
		z-index: 101;
	} */
</style>
