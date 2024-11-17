<script lang="ts">
    import { onMount } from 'svelte';
    import * as d3 from 'd3';
    
    let { coords, width, height, onHover }: {
        coords: number[][];
        width: number;
        height: number;
        onHover: (event: MouseEvent, point: { x: number, y: number, index: number }) => void;
    } = $props();

    let svg: d3.Selection<SVGGElement, unknown, null, undefined>;
    let container: HTMLElement;
    
    // Add padding to avoid points being cut off
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create scales for x and y
    const xScale = d3.scaleLinear()
        .domain(d3.extent(coords, d => d[0]) as [number, number])
        .range([0, innerWidth])
        .nice();

    const yScale = d3.scaleLinear()
        .domain(d3.extent(coords, d => d[1]) as [number, number])
        .range([innerHeight, 0])
        .nice();

    onMount(() => {
        // Create SVG container
        svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Add X axis
        svg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${innerHeight})`)
            .call(d3.axisBottom(xScale))
            .call(g => g.select('.domain').attr('stroke', '#ccc'))
            .call(g => g.selectAll('.tick line').attr('stroke', '#ccc'));

        // Add Y axis
        svg.append('g')
            .attr('class', 'y-axis')
            .call(d3.axisLeft(yScale))
            .call(g => g.select('.domain').attr('stroke', '#ccc'))
            .call(g => g.selectAll('.tick line').attr('stroke', '#ccc'));

        // Add grid lines
        svg.append('g')
            .attr('class', 'grid-lines')
            .selectAll('line')
            .data(yScale.ticks())
            .join('line')
            .attr('x1', 0)
            .attr('x2', innerWidth)
            .attr('y1', d => yScale(d))
            .attr('y2', d => yScale(d))
            .attr('stroke', '#f0f0f0')
            .attr('stroke-width', 0.5);

        // Add points
        svg.selectAll('circle')
            .data(coords)
            .join('circle')
            .attr('cx', d => xScale(d[0]))
            .attr('cy', d => yScale(d[1]))
            .attr('r', 5)
            .attr('class', 'point')
            .on('mouseover', (event, d) => {
                d3.select(event.currentTarget)
                    .transition()
                    .duration(200)
                    .attr('r', 8);
                onHover(event, { x: d[0], y: d[1], index: coords.indexOf(d) });
            })
            .on('mouseout', (event) => {
                d3.select(event.currentTarget)
                    .transition()
                    .duration(200)
                    .attr('r', 5);
                onHover(event, { x: -1, y: -1, index: -1 });
            });
    });
</script>

<div 
    bind:this={container} 
    class="scatter-plot-container"
>
</div>

<style>
    .scatter-plot-container {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem;
    }

    :global(.scatter-plot-container .point) {
        fill: #6366f1;
        fill-opacity: 0.6;
        stroke: #4f46e5;
        stroke-width: 1;
        transition: fill-opacity 0.2s ease;
        cursor: pointer;
    }

    :global(.scatter-plot-container .point:hover) {
        fill-opacity: 0.8;
    }

    :global(.scatter-plot-container .x-axis text),
    :global(.scatter-plot-container .y-axis text) {
        font-size: 12px;
        color: #666;
    }

    :global(.scatter-plot-container .x-axis path),
    :global(.scatter-plot-container .y-axis path) {
        stroke-width: 1;
    }

    :global(.scatter-plot-container .x-axis line),
    :global(.scatter-plot-container .y-axis line) {
        stroke-width: 1;
    }
</style>