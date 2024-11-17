<script lang="ts">
	import {
		getBezierPath,
		getSmoothStepPath,
		type EdgeProps,
		EdgeLabelRenderer,
		useEdges,
		SmoothStepEdge
	} from '@xyflow/svelte';
	
	type $$Props = EdgeProps;

	const {
		id,
		sourceX,
		sourceY,
		sourcePosition,
		targetX,
		targetY,
		targetPosition,
		markerEnd,
		style,
		label
	}: $$Props = $props();
	
	const edgeInterpolation = getSmoothStepPath;
	
	const [edgePath, labelX, labelY] = $derived(edgeInterpolation({
		sourceX,
		sourceY,
		sourcePosition,
		targetX,
		targetY,
		targetPosition
	}));
	
	// if (id.includes('window')) {
	// 	console.log(id);
	// }

	const edges = useEdges();

	const onEdgeClick = () => {
		// Remove the edge
		// edges.update((eds) => eds.filter((edge) => edge.id !== id));
	};
</script>

<SmoothStepEdge {style} {markerEnd} {sourceX} {sourceY} {sourcePosition} {targetX} {targetY} {targetPosition} />

<style>
</style>
