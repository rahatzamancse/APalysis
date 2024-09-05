import React from 'react'
import { Node } from '../types'
import * as api from '../api'
import { findIndicesOfMax, transposeArray } from '../utils'
import * as d3 from 'd3'
import ImageToolTip from './ImageToolTip'

function NodeActivationMatrix({ node, width, height }: { node: Node, width: number, height: number }) {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const svgRef = React.useRef<SVGSVGElement>(null)
    const [hoveredItem, setHoveredItem] = React.useState<[number, number]>([-1, -1])
    const [nExamples, setNExamples] = React.useState<number>(0)

    const svgPadding = { top: 10, right: 10, bottom: 10, left: 10 }
    
    React.useEffect(() => {
        if (['Conv2D', 'Concatenate', 'Dense', 'Conv2d', 'Cat', 'Linear', 'Add'].some(l => node.layer_type.includes(l))) {
            api.getAnalysisHeatmap(node.name).then(setHeatmap)
        }
        api.getTotalInputs().then(setNExamples)
    }, [node.layer_type, node.name])

    if (heatmap.length === 0) return null
        
    // Normalize all rows in heatmap
    const finalHeatmap = transposeArray(transposeArray(heatmap).map(row => {
        // Mean shift
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const meanShiftedRow = row.map(item => item - mean)
        
        // Normalize
        const max = Math.max(...meanShiftedRow)
        const min = Math.min(...meanShiftedRow)
        const normalRow = meanShiftedRow.map(item => (item - min) / (max - min))
        return normalRow
    }))

    // Choose only first TOTAL_MAX_CHANNELS elements of each row
    const TOTAL_MAX_CHANNELS = (arr: number[]) => arr.length * 0.5
    // const TOTAL_MAX_CHANNELS = (arr: number[]) => 10
    const indicesMax = finalHeatmap.map(arr => findIndicesOfMax(arr, TOTAL_MAX_CHANNELS(arr)))
    finalHeatmap.forEach((col, i) => {
        // Make all other elements of arr 0 except the max elements
        col.forEach((_, j) => {
            if(!indicesMax[i].includes(j)){
                col[j] = 0
            }
        })
    })
    
    // Calculate Jaccard similarity between each pair of columns
    const jDist = finalHeatmap.map((col1, i) => {
        return finalHeatmap.map((col2, j) => {
            if (i === j) return [1, 1, 1]
            const intersection = col1
                .map((item, k) => (item > 0 && col2[k] > 0))
                .reduce((total, x) => total + (x?1:0), 0)
            const union = col1
                .map((item, k) => (item > 0 || col2[k] > 0))
                .reduce((total, x) => total + (x?1:0), 0)
            return [intersection / union, intersection, union]
        })
    })
    
    // Get maximum and minimum from jDist
    const maxJDist = Math.max(...jDist.map(col => Math.max(...col.map(item => item[0]))))
    const minJDist = Math.min(...jDist.map(col => Math.min(...col.map(item => item[0]))))
    
    // Get maximum but skip 1s
    const maxJDistWithout1 = Math.max(...jDist.map(col => Math.max(...col.map(item => item[0] === 1?0:item[0]))))

    // Colorscale for jDist heatmap
    const jDistColorScale = d3.scaleLinear()
        .domain([minJDist, maxJDistWithout1])
        .range([0, 1])
        .clamp(true)

    const jDistColors = jDist.map(col => col.map(item => d3.interpolateBlues(jDistColorScale(item[0]))))
    
    const cellWidth = (width - svgPadding.left - svgPadding.right) / nExamples
    const cellHeight = (height - svgPadding.top - svgPadding.bottom) / nExamples

    return <>
        <svg width={width+20} height={height+20} ref={svgRef} style={{
            backgroundColor: "white"
        }}>
            <defs>
                <pattern id="diagonalHatch" patternUnits="userSpaceOnUse" width="4" height="4">
                    <path d="M-1,1 l2,-2
                            M0,4 l4,-4
                            M3,5 l2,-2"
                        style={{stroke:'black', strokeWidth:1}}
                    />
                </pattern>
            </defs>
            <g transform='translate(20 20)'>
                {jDistColors.map((col, i) => col.map((elem, j) => (
                    <rect
                        key={`${i}-${j}`}
                        x={i * cellWidth + svgPadding.left}
                        y={j * cellHeight + svgPadding.top}
                        width={cellWidth}
                        height={cellHeight}
                        fill={i === j?"url(#diagonalHatch)":elem}
                        onMouseEnter={() => {
                            setHoveredItem([i,j])
                        }}
                        onMouseLeave={() => {
                            setHoveredItem([-1, -1])
                        }}
                        data-tooltip-id="image-tooltip"
                    />
                )))}
            </g>
        </svg>
        {hoveredItem[0] !== -1 && <ImageToolTip
            imgs={hoveredItem}
            imgType={'raw'}
            imgData={{}}
            label={`Jaccard similarity: ${jDist[hoveredItem[0]][hoveredItem[1]][1]}/${jDist[hoveredItem[0]][hoveredItem[1]][2]}`}
        />}
    </>
}

export default NodeActivationMatrix