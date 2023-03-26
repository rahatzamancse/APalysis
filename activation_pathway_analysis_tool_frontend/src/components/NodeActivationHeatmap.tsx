import React from 'react'
import * as api from '../api'
import { Node } from '../types'
import * as d3 from 'd3'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice'
import { chunkify } from '../utils'

function transposeArray<T>(array: T[][]): T[][] {
    return array[0].map((_, j) =>
        array.map((row) => row[j])
    );
}
function findIndicesOfMax(inp: number[], count: number) {
    var outp = [];
    for (var i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function(a, b) { return inp[b] - inp[a]; }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}
function calcAllPairwiseDistance(arr: number[]) {
    let sum = 0
    for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            sum += Math.abs(arr[i] - arr[j])
        }
    }
    return sum
}

function calcVariance(inp: number[]) {
    const mean = inp.reduce((a, b) => a + b, 0) / inp.length
    const variance = inp.map(item => Math.pow(item - mean, 2)).reduce((a, b) => a + b, 0) / inp.length
    return variance
}
function calcPairwiseDistance(arr1: number[], arr2: number[]) {
    let sum = 0
    for (let i = 0; i < arr1.length; i++) {
        sum += Math.pow(arr1[i] - arr2[i], 2)
    }
    return Math.sqrt(sum)
}

function calcSumPairwiseDistance(...arrs: number[][]): number {
    let sum = 0
    for (let i = 0; i < arrs.length; i++) {
        for (let j = i + 1; j < arrs.length; j++) {
            sum += calcPairwiseDistance(arrs[i], arrs[j])
        }
    }
    return sum
}

function getRawHeatmap(heatmap: number[][], nExamples: number, nClasses: number) {
    return heatmap.slice(0, nExamples)
}

function NodeActivationHeatmap({ node, width, height }: { node: Node, width: number, height: number }) {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const svgRef = React.useRef<SVGSVGElement>(null)
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const [globalColorScale, setGlobalColorScale] = React.useState(false)
    const [hoveredRow, setHoveredRow] = React.useState<number>(-1)

    const svgPadding = { top: 10, right: 10, bottom: 10, left: 10 }

    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        if (!['conv', 'mixed'].some(l => node.name.toLowerCase().includes(l))) return
        api.getAnalysisHeatmap(node.name).then(setHeatmap)
    }, [node.name, analyzeResult])

    if (heatmap.length === 0) return null
    const nExamples = analyzeResult.examplePerClass * analyzeResult.selectedClasses.length
        
    // Normalize all rows in heatmap
    const normalHeatmap = transposeArray(transposeArray(heatmap).map(row => {
        // Mean shift
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const meanShiftedRow = row.map(item => item - mean)
        
        // Normalize
        const max = Math.max(...meanShiftedRow)
        const min = Math.min(...meanShiftedRow)
        const normalRow = meanShiftedRow.map(item => (item - min) / (max - min))
        return normalRow
    }))

    // Add variance as a new column for each row
    const h1 = transposeArray(transposeArray(
        getRawHeatmap(normalHeatmap, nExamples, analyzeResult.selectedClasses.length))
            .map(row => [...row, calcVariance(row)])
        )

    const h2 = transposeArray(
        transposeArray(h1).map(row => [...row, (analyzeResult.selectedClasses.length>1?calcSumPairwiseDistance:calcAllPairwiseDistance)(
            ...chunkify(row.slice(0, nExamples), analyzeResult.examplePerClass)
        )])
    )

    // Choose only first TOTAL_MAX_CHANNELS elements of each row
    const TOTAL_MAX_CHANNELS = (arr: number[]) => arr.length * 0.2
    // const TOTAL_MAX_CHANNELS = (arr: number[]) => 10
    const indicesMax = h2.map(arr => findIndicesOfMax(arr, TOTAL_MAX_CHANNELS(arr)))
    h2.forEach((col, i) => {
        // Make all other elements of arr 0 except the max elements
        col.forEach((_, j) => {
            if(!indicesMax[i].includes(j)){
                col[j] = 0
            }
        })
    })
    
    
    // Calculate different between number of activated channels and number of examples
    const h3 = transposeArray(
        transposeArray(h2).map(row => [
            ...row,
            calcAllPairwiseDistance(
                analyzeResult.selectedClasses.length>1?
                    chunkify(row.slice(0, nExamples), analyzeResult.examplePerClass)
                        .map(row => row.reduce((prev, curr) => prev + (curr > 0?1:0), 0)):
                    row.slice(0, nExamples)
            )
        ])
    )
    
    // Sort all colors by the last summary columns
    // 1: if there are multiple classes: Sum of pairwise activation count (after taking TOTAL_MAX_CHANNELS) distance between all classes
    // 1: if there is only one class: Sum of pairwise activation value (after making all values 0 except TOTAL_MAX_CHANNELS) distance between all examples
    // 2: if there are multiple classes: Sum of pairwise activation value distance between all classes
    // 2: if there is only one class: Sum of pairwise activation value distance between all examples
    // 3: Variance of all examples
    const SORT_BY = 1 
    const finalHeatmap = transposeArray(transposeArray(h3).sort((a, b) => b[b.length - SORT_BY] - a[a.length - SORT_BY]))
    
    
    // Apply the colorScale to finalHeatmap
    const colorScales = finalHeatmap.map(col => d3.scaleLinear<number>()
        .domain(globalColorScale?[
            Math.min(...finalHeatmap.map(x => Math.min(...x))),
            Math.max(...finalHeatmap.map(x => Math.max(...x))),
        ]:[
            Math.min(...col),
            Math.max(...col),
        ])
        .range([0, 1])
        .clamp(true)
    )
    const allColors = finalHeatmap.map((col, i) => col.map(elem => colorScales[i](elem)))

    // Apply the color scale to allColors
    const heatmapColor = transposeArray(transposeArray(allColors)).map(row => row.map(d3.interpolateBlues))
    const extraCols = heatmapColor.length - analyzeResult.selectedClasses.length*analyzeResult.examplePerClass
    
    // Drawing parameters
    const cellWidth = (width - svgPadding.left - svgPadding.right) / heatmapColor.length
    const cellHeight = (height - svgPadding.top - svgPadding.bottom) / heatmapColor[0].length
    const labelScale = d3.scaleLinear()
        .domain([0, analyzeResult.selectedClasses.length - 1])
        .range([
            svgPadding.left + (cellWidth * analyzeResult.examplePerClass) / 2,
            width - svgPadding.right - (cellWidth * analyzeResult.examplePerClass) / 2 - cellWidth*extraCols
        ])
        

    return <>
        <svg width={width} height={height} ref={svgRef} style={{
            backgroundColor: "white"
        }}>
            <g>
                {heatmapColor.map((col, i) => col.map((elem, j) => (
                    <rect
                        key={`${i}-${j}`}
                        x={i * cellWidth + svgPadding.left}
                        y={j * cellHeight + svgPadding.top}
                        width={cellWidth}
                        height={cellHeight}
                        fill={elem}
                        onMouseEnter={() => {
                            if(cellHeight > 5)
                                setHoveredRow(j)
                        }}
                        onMouseLeave={() => {
                            setHoveredRow(-1)
                        }}
                    />
                )))}
            </g>
            <g>
                <line
                    x1={svgPadding.left}
                    y1={height - svgPadding.bottom}
                    x2={width}
                    y2={height - svgPadding.bottom}
                    stroke="black"
                />
                <line
                    x1={svgPadding.left}
                    y1={height - svgPadding.bottom}
                    x2={svgPadding.left}
                    y2={0}
                    stroke="black"
                />
                <line
                    x1={width - svgPadding.right - cellWidth*extraCols}
                    y1={height - svgPadding.bottom}
                    x2={width - svgPadding.right - cellWidth*extraCols}
                    y2={0}
                    stroke="black"
                />
                <text transform={`translate(${width / 2}, ${height - 1})`}>
                    Images
                </text>
                <text
                    textAnchor='middle'
                    transform={`translate(${0}, ${height / 2}) rotate(90)`}>
                    Channel Activation
                </text>
                {/* Add a line seperator between each class */}
                {Array.from({ length: analyzeResult.selectedClasses.length-1 }, (_, i) => (
                    <line
                        key={i}
                        x1={labelScale(i + 1) - (cellWidth * analyzeResult.examplePerClass) / 2}
                        y1={svgPadding.top}
                        x2={labelScale(i + 1) - (cellWidth * analyzeResult.examplePerClass) / 2}
                        y2={height - svgPadding.bottom}
                        stroke="black"
                    />
                ))}
            </g>
            <g>
                {/* Add title for each class */}
                {analyzeResult.selectedClasses.map((label, i) => (
                    <text key={i}
                        textAnchor='middle'
                        transform={`translate(${labelScale(i)}, ${svgPadding.top})`}
                    >
                        {label}
                    </text>
                ))}
            </g>
            <g>
                {/* Add a highlighting line above and below the rects of hoveredRow */}
                {hoveredRow !== -1 && (
                    <>
                        {/* y={j * cellHeight + svgPadding.top} */}
                        <line
                            x1={svgPadding.left}
                            y1={svgPadding.top + hoveredRow * cellHeight}
                            x2={width - svgPadding.right}
                            y2={svgPadding.top + hoveredRow * cellHeight}
                            stroke="yellow"
                        />
                        <line
                            x1={svgPadding.left}
                            y1={svgPadding.top + (hoveredRow + 1) * cellHeight}
                            x2={width - svgPadding.right}
                            y2={svgPadding.top + (hoveredRow + 1) * cellHeight}
                            stroke="yellow"
                        />
                    </>
                )}
            </g>
        </svg>
    </>
}

export default NodeActivationHeatmap