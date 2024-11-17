<script lang="ts">
	let open = $state(false);
	import { slide } from 'svelte/transition';
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	
	type Props = HTMLAttributes<HTMLDivElement> & {
		header: Snippet,
		details: Snippet,
		alwaysOpen?: boolean,
	}

	const { header, details, alwaysOpen = false }: Props = $props();
	if (alwaysOpen) open = true;

	const handleClick = () => (open = !open);
</script>

<div class="accordion">
	<div class="header">
		<div class="text">
			{@render header()}
		</div>

		<button onclick={handleClick}> +/- </button>
	</div>

	{#if alwaysOpen || open}
		<div class="details" transition:slide>
			{@render details()}
		</div>
	{/if}
</div>

<style>
	div.accordion {
		margin: 1rem 0;
	}

	div.header {
		display: flex;
		width: 100%;
	}

	div.header .text {
		flex: 1;
		margin-right: 5px;
	}

	div.details {
		background-color: #cecece;
		padding: 1rem;
	}
</style>
