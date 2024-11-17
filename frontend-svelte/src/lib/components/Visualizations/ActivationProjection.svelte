<script lang="ts">
	import * as api from '$lib/api';
	import ScatterPlot from '$lib/components/Visualizations/ScatterPlot.svelte';
	import * as Popover from '$lib/components/ui/popover/index.js';

	let {
		tensorId, shape
	}: {
		tensorId: string;
		shape: number[];
	} = $props();

	let coords = $state<[number, number][]>([]);
	let distances = $state<number[][]>([]);
	let isPopupOpen = $state(false);
	
	function checkIfImage(shape: number[]) {
		return shape.length === 3 && shape[0] === 3
	}

	let popupData = $state<string | null | number[]>(null);
	let popupDataType = $state<'image' | 'numpy'>(checkIfImage(shape) ? 'image' : 'numpy');

	let mousePosition = $state<{ x: number; y: number }>({ x: 0, y: 0 });

	$effect(() => {
		api.getProjection(tensorId.toString(), 'pca', 'euclidean', 'none').then((c) => (coords = c));
	});
	

	function onHover(event: MouseEvent, point: { x: number; y: number; index: number }) {
		mousePosition = { x: event.clientX, y: event.clientY };
		
		if (point.index === -1) {
			popupData = null;
			isPopupOpen = false;
			return;
		}
		api.getInputShape(point.index).then((shape) => {
			popupDataType = checkIfImage(shape) ? 'image' : 'numpy';
			if (popupDataType === 'image') {
				api.getInputImage(point.index).then((input) => {
					popupData = input;
					isPopupOpen = true;
				});
			}
			else {
				popupData = shape;
				isPopupOpen = true;
			}
		});
	}
</script>

<div>
	{#if coords.length === 0}
		<div class="loading">
			<div class="spinner"></div>
			<p>Loading projection...</p>
		</div>
	{:else}
		<ScatterPlot {coords} {onHover} width={200} height={200} />
		<Popover.Root bind:open={isPopupOpen}>
			<Popover.Trigger class="hidden">About</Popover.Trigger>
			<Popover.Content class="w-80" style="position: fixed; left: {mousePosition.x}px; top: {mousePosition.y}px;">
				<div class="grid gap-4">
					{#if popupDataType === 'numpy'}
						{JSON.stringify(popupData)}
					{:else}
						<img src={popupData as string} alt="Input representing this point" />
					{/if}
				</div>
			</Popover.Content>
		</Popover.Root>
	{/if}
</div>

<style>
</style>
