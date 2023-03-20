import React from 'react'
import * as api from '../api'
import { Node } from '../types'
import * as d3 from 'd3'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice'
import Form from 'react-bootstrap/Form';

function transposeArray<T>(array: T[][]): T[][] {
    const numRows = array.length;
    const numCols = array[0].length;

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
function calcVariance(inp: number[]) {
    const mean = inp.reduce((a, b) => a + b, 0) / inp.length
    const variance = inp.map(item => Math.pow(item - mean, 2)).reduce((a, b) => a + b, 0) / inp.length
    return variance
}

function NodeActivationHeatmap({ node, width, height }: { node: Node, width: number, height: number }) {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const svgRef = React.useRef<SVGSVGElement>(null)
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const [globalColorScale, setGlobalColorScale] = React.useState(false)

    const svgPadding = { top: 10, right: 10, bottom: 10, left: 10 }

    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        if (!['conv', 'mixed'].some(l => node.name.toLowerCase().includes(l))) return
        api.getAnalysisHeatmap(node.name).then(setHeatmap)
    }, [node.name, analyzeResult])

    if (heatmap.length === 0) return null
        
    // Normalize all rows in heatmap
    const normalHeatmap = heatmap.map(row => {
        // Mean shift
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const meanShiftedRow = row.map(item => item - mean)
        
        // Normalize
        const max = Math.max(...meanShiftedRow)
        const min = Math.min(...meanShiftedRow)
        const normalRow = meanShiftedRow.map(item => (item - min) / (max - min))
        return normalRow
    })

    // Add variance as a new column for each row
    const normalHeatmapWithVariance = transposeArray(transposeArray(normalHeatmap).map(row => [...row, calcVariance(row)]))
    
    const TOTAL_MAX_CHANNELS = (arr: number[]) => arr.length * 0.1
    // const TOTAL_MAX_CHANNELS = (arr: number[]) => 10
    
    const colorScales = normalHeatmapWithVariance.map(row => d3.scaleLinear<number>()
        .domain(globalColorScale?[
            Math.min(...normalHeatmapWithVariance.map(x => Math.min(...x))),
            Math.max(...normalHeatmapWithVariance.map(x => Math.max(...x))),
        ]:[
            Math.min(...row),
            Math.max(...row),
        ])
        .range([0, 1])
        // .range(["red", "blue"])
        .clamp(true)
    )

    // Apply the colorScale to normalHeatmapWithVariance
    const cellWidth = normalHeatmapWithVariance.length !== 0 ? (width - svgPadding.left - svgPadding.right) / normalHeatmapWithVariance.length : 0
    const cellHeight = normalHeatmapWithVariance.length !== 0 ? (height - svgPadding.top - svgPadding.bottom) / normalHeatmapWithVariance[0].length : 0
   
    const heatmapColorT = transposeArray(normalHeatmapWithVariance.map((row, i) => row.map(col => colorScales[i](col))))

    const allColors = transposeArray(heatmapColorT)
    const indicesMax = allColors.map(arr => findIndicesOfMax(arr, TOTAL_MAX_CHANNELS(arr)))
    allColors.forEach((arr, i) => {
        // Make all other elements of arr 0 except the max elements
        arr.forEach((_, j) => {
            if(!indicesMax[i].includes(j)){
                arr[j] = 0
            }
        })
    })
    const heatmapColor = transposeArray(transposeArray(allColors)).map(row => row.map(d3.interpolateBlues))

    const groupedLabels = analyzeResult.selectedClasses

    const labelScale = d3.scaleLinear()
        .domain([0, analyzeResult.selectedClasses.length - 1])
        .range([
            svgPadding.left + (cellWidth * analyzeResult.examplePerClass) / 2,
            width - svgPadding.right - (cellWidth * analyzeResult.examplePerClass) / 2 - (cellWidth)
        ])

    return <>
        <svg width={width} height={height} ref={svgRef} style={{
            backgroundColor: "white"
        }}>
            <g>
                {heatmapColor.map((row, i) => row.map((col, j) => (
                    <rect key={`${i}-${j}`} x={i * cellWidth + svgPadding.left} y={j * cellHeight + svgPadding.top} width={cellWidth} height={cellHeight} fill={col} />
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
                    x1={width - svgPadding.right - cellWidth}
                    y1={height - svgPadding.bottom}
                    x2={width - svgPadding.right - cellWidth}
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
                {/* Add a line seperator between each rect */}
                {Array.from({ length: analyzeResult.selectedClasses.length - 1 }, (_, i) => (
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
                {groupedLabels.map((label, i) => (
                    <text key={i}
                        textAnchor='middle'
                        transform={`translate(${labelScale(i)}, ${svgPadding.top})`}
                    >
                        {label}
                    </text>
                ))}
            </g>
        </svg>
    </>
}

export default NodeActivationHeatmap