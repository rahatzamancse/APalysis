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
function sumPairwiseDistance(arr: number[]): number {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            const distance = Math.abs(arr[i] - arr[j]);
            sum += distance;
        }
    }
    return sum;
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
        

    
    const colorScales = heatmap.map(row => d3.scaleLinear<number>()
        .domain(globalColorScale?[
            Math.min(...heatmap.map(x => Math.min(...x))),
            Math.max(...heatmap.map(x => Math.max(...x))),
        ]:[
            Math.min(...row),
            Math.max(...row),
        ])
        .range([0, 1])
        // .range(["red", "blue"])
        .clamp(true)
    )

    // Apply the colorScale to heatmap
    const cellWidth = heatmap.length !== 0 ? (width - svgPadding.left - svgPadding.right) / heatmap.length : 0
    const cellHeight = heatmap.length !== 0 ? (height - svgPadding.top - svgPadding.bottom) / heatmap[0].length : 0
   
    const heatmapColorT = transposeArray(heatmap.map((row, i) => row.map(col => colorScales[i](col))))

    // Similarity Ranking
    const summary = heatmapColorT.map(
        // Group by class
        (row, i) => sumPairwiseDistance(Array.from({ length: Math.ceil(row.length / analyzeResult.examplePerClass) }, (_, j) =>
            row.slice(j * analyzeResult.examplePerClass, (j + 1) * analyzeResult.examplePerClass)
        // Reduce each class group to average
        ).map(classGroup => classGroup.reduce((a, b) => a + b, 0) / classGroup.length))
    )
    
    const ranking = summary.map((_, i) => i).sort((a, b) => summary[a] - summary[b])
        
    if(node.name == 'block1_conv1') {
        console.log(summary)
    }
    
    heatmapColorT.sort((a, b) => ranking[heatmapColorT.indexOf(a)] - ranking[heatmapColorT.indexOf(b)])
    const heatmapColor = transposeArray(heatmapColorT).map(row => row.map(d3.interpolateBlues))

    // d3.interpolateBlues()
        
    // const heatmapColor = heatmap.map((row, i) => row.map(col => colorScales[i](col)))

    const groupedLabels = analyzeResult.selectedClasses

    const labelScale = d3.scaleLinear()
        .domain([0, analyzeResult.selectedClasses.length - 1])
        .range([
            svgPadding.left + (cellWidth * analyzeResult.examplePerClass) / 2,
            width - svgPadding.right - (cellWidth * analyzeResult.examplePerClass) / 2
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