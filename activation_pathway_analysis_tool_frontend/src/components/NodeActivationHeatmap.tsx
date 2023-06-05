import React, { FC } from 'react'
import * as api from '../api'
import { Node } from '../types'
import * as d3 from 'd3'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice'
import { calcAllPairwiseDistance, calcSumPairwiseDistance, calcVariance, chunkify, findIndicesOfMax, getRawHeatmap, transposeArray } from '../utils'
import ImageToolTip from './ImageToolTip'

interface Props {
    node: Node;
    width: number;
    height: number;
    normalizeRow?: boolean;
    sortby?: 'count' | 'pairwise' | 'variance' | 'none' | 'edge_weight';
    totalMaxChannels?: (arr: number[]) => number;
}

const NodeActivationHeatmap: FC<Props> = ({ node, width, height, normalizeRow, sortby, totalMaxChannels }) => {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const svgRef = React.useRef<SVGSVGElement>(null)
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const [globalColorScale, setGlobalColorScale] = React.useState(false)
    const [hoveredItem, setHoveredItem] = React.useState<[number, number]>([-1, -1])
    const [sortBy, setSortBy] = React.useState<'count' | 'pairwise' | 'variance' | 'edge_weight' | 'none'>(sortby!)

    const svgPadding = { top: 10, right: 10, bottom: 10, left: 10 }

    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        if (['Conv2D', 'Concatenate', 'Dense'].some(l => node.layer_type.includes(l))) {
            api.getAnalysisHeatmap(node.name).then(setHeatmap)
        }
    }, [node.name, analyzeResult])
    
    if (heatmap.length === 0) return null
    const nExamples = analyzeResult.examplePerClass * analyzeResult.selectedClasses.length
        
    // Normalize all rows in heatmap
    const normalHeatmap = normalizeRow?transposeArray(transposeArray(heatmap).map(row => {
        // Mean shift
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const meanShiftedRow = row.map(item => item - mean)
        
        // Normalize
        const max = Math.max(...meanShiftedRow)
        const min = Math.min(...meanShiftedRow)
        const normalRow = (max - min === 0)?meanShiftedRow.map(item => 0):meanShiftedRow.map(item => (item - min) / (max - min))
        return normalRow
    })):heatmap.map(col => {
        // Mean shift
        const mean = col.reduce((a, b) => a + b, 0) / col.length
        const meanShiftedCol = col.map(item => item - mean)

        // Normalize
        const max = Math.max(...meanShiftedCol)
        const min = Math.min(...meanShiftedCol)
        const normalCol = (max - min === 0)?meanShiftedCol.map(item => 0):meanShiftedCol.map(item => (item - min) / (max - min))
        return normalCol
    })

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

    // Choose only first totalMaxChannels elements of each row
    const indicesMax = h2.map(arr => findIndicesOfMax(arr, totalMaxChannels!(arr)))
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
    
    let h4
    if(['Conv2D'].includes(node.layer_type)) {
        h4 = transposeArray(
            transposeArray(h3).map((row, i) => [
                ...row,
                node.out_edge_weight[i]
            ])
        )
    } else {
        h4 = h3
    }
    
    // Sort all colors by the last summary columns
    // 0: Keep the original order
    // 1: if there are multiple classes: Sum of pairwise activation count (after taking totalMaxChannels) distance between all classes
    // 1: if there is only one class: Sum of pairwise activation value (after making all values 0 except totalMaxChannels) distance between all examples
    // 2: if there are multiple classes: Sum of pairwise activation value distance between all classes
    // 2: if there is only one class: Sum of pairwise activation value distance between all examples
    // 3: Variance of all examples
    let SORT_BY = 0
    if(sortBy === 'count') SORT_BY = 2
    else if(sortBy === 'edge_weight') SORT_BY = 1
    else if(sortBy === 'pairwise') SORT_BY = 3
    else if(sortBy === 'variance') SORT_BY = 4
     
    const finalHeatmap = SORT_BY === 0 ? h4:transposeArray(transposeArray(h4).sort((a, b) => b[b.length - SORT_BY] - a[a.length - SORT_BY]))
    
    
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
                            if (['Conv2D', 'Concatenate'].some(l => node.layer_type.includes(l)))
                                setHoveredItem([i, j])
                        }}
                        onMouseLeave={() => {
                            if (['Conv2D', 'Concatenate'].some(l => node.layer_type.includes(l)))
                                setHoveredItem([-1, -1])
                        }}
                        data-tooltip-id="image-tooltip"
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
                {/* Add a highlighting line above and below the rects of hoveredItem */}
                {/* {hoveredItem[1] !== -1 && (
                    <>
                        <line
                            x1={svgPadding.left}
                            y1={svgPadding.top + hoveredItem[1] * cellHeight}
                            x2={width - svgPadding.right}
                            y2={svgPadding.top + hoveredItem[1] * cellHeight}
                            stroke="yellow"
                        />
                        <line
                            x1={svgPadding.left}
                            y1={svgPadding.top + (hoveredItem[1] + 1) * cellHeight}
                            x2={width - svgPadding.right}
                            y2={svgPadding.top + (hoveredItem[1] + 1) * cellHeight}
                            stroke="yellow"
                        />
                    </>
                )} */}
            </g>
        </svg>
        {hoveredItem[0] !== -1 && hoveredItem[0] < nExamples && <ImageToolTip
            imgs={[hoveredItem[0]]}
            imgType={'overlay'}
            imgData={{
                layer: node.name,
                channel: hoveredItem[1]
            }}
        />}
    </>
}

NodeActivationHeatmap.defaultProps = {
    normalizeRow: true,
    sortby: 'count',
    totalMaxChannels: arr => arr.length * 0.2,
}


export default NodeActivationHeatmap