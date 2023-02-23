import React from 'react'
import * as api from '../api'
import { Node } from '../types'
import * as d3 from 'd3'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice'
import Form from 'react-bootstrap/Form';

function linSpace(startValue: number, stopValue: number, cardinality: number) {
    var arr = [];
    var step = (stopValue - startValue) / (cardinality - 1);
    for (var i = 0; i < cardinality; i++) {
        arr.push(startValue + (step * i));
    }
    return arr;
}

function NodeActivationHeatmap({ node, width, height }: { node: Node, width: number, height: number }) {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const svgRef = React.useRef<SVGSVGElement>(null)
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const [globalColorScale, setGlobalColorScale] = React.useState(true)

    const svgPadding = { top: 10, right: 10, bottom: 10, left: 10 }

    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        if (!['conv', 'mixed'].some(l => node.name.toLowerCase().includes(l))) return
        api.getAnalysisHeatmap(node.name).then(setHeatmap)
    }, [node.name, analyzeResult])

    if (heatmap.length === 0) return null

    const colorScales = heatmap.map(row => d3.scaleLinear()
        .domain(globalColorScale?[
            Math.min(...heatmap.map(x => Math.min(...x))),
            Math.max(...heatmap.map(x => Math.max(...x))),
        ]:[
            Math.min(...row),
            Math.max(...row),
        ])
        .range([0, 1])
        .clamp(true)
    )

    // Apply the colorScale to heatmap
    const cellWidth = heatmap.length !== 0 ? (width - svgPadding.left - svgPadding.right) / heatmap.length : 0
    const cellHeight = heatmap.length !== 0 ? (height - svgPadding.top - svgPadding.bottom) / heatmap[0].length : 0
    const heatmapColor = heatmap.map((row, i) => row.map(col => d3.interpolateBlues(colorScales[i](col))))

    const groupedLabels = analyzeResult.selectedClasses

    const labelScale = d3.scaleLinear()
        .domain([0, analyzeResult.selectedClasses.length - 1])
        .range([
            svgPadding.left + (cellWidth * analyzeResult.examplePerClass) / 2,
            width - svgPadding.right - (cellWidth * analyzeResult.examplePerClass) / 2
        ])

    return <>
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
        }}>
            <h5 style={{}}>Activation Heatmap</h5>
            <Form.Check
                type="switch"
                id="custom-switch"
                label="Global Color Scale"
                checked={globalColorScale}
                onChange={e => setGlobalColorScale(e.target.checked)}
            />
        </div>
        <svg width={width} height={height} ref={svgRef}>
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