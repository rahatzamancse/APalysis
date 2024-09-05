import React from 'react';
import { useAppSelector } from '../app/hooks';
import { selectAnalysisResult } from '../features/analyzeSlice';
import * as d3 from 'd3';
import ImageToolTip from './ImageToolTip'
import { Node } from '../types';
import * as api from '../api';
import smoothHull from '../convexHull';

type Point = [number, number];
const x = (d: Point) => d[0];
const y = (d: Point) => d[1];

const svgMargin = 10

function ScatterPlot({ node, coords, preds, distances, labels, width, height }: {
  coords: Point[];
  preds: boolean[];
  distances: number[][];
  labels: number[];
  width: number;
  height: number;
  node: Node | null;
}) {
  const [showLines, setShowLines] = React.useState<boolean>(false)
  const analysisResult = useAppSelector(selectAnalysisResult)
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [hoveredItem, setHoveredItem] = React.useState<number>(-1)
  const [clusterPaths, setClusterPaths] = React.useState<string[]>([])
  const xScale = React.useMemo(
    () => d3.scaleLinear()
      .domain([Math.min(...coords.map(x)), Math.max(...coords.map(x))])
      .range([svgMargin, width - svgMargin])
      .clamp(true),
    [width, coords],
  );
  const yScale = React.useMemo(
    () =>
      d3.scaleLinear()
        .domain([Math.min(...coords.map(y)), Math.max(...coords.map(y))])
        .range([height - svgMargin, svgMargin])
        .clamp(true),
    [height, coords],
  );
  const colorScale = React.useMemo(
    () => d3.scaleOrdinal<string>()
      .domain(Array.from(new Set(labels)).map((d) => d.toString()))
      .range(d3.schemeCategory10.slice()),
    [labels]
  );
  
  const opacityScale = React.useMemo(
    () => d3.scaleLinear()
      .domain([
        Math.min(...distances.map((row) => Math.min(...row.filter(d => d !== 0 && isFinite(d))))),
        Math.max(...distances.map((row) => Math.max(...row.filter(d => d !== 0 && isFinite(d))))),
      ])
      .range([1, 0]),
    [distances]
  )

  // const topDistanceIndices = React.useMemo(() => {
  //   const topDistances = distances.map((row) => row.slice().sort((a, b) => a - b).slice(1, 4))
  //   const topDistanceIndices = topDistances.map((row, i) => row.map((d) => distances[i].indexOf(d)))
  //   return topDistanceIndices
  // }, [distances])
  // 

  

  
  return (
    <div>
      <div>
        {node && <button type="button" style={{ float: 'right' }} className="btn btn-primary" onClick={(e) => {
          api.getCluster(node.name).then((clusters) => {
            if(clusters.labels.length > 0) {
              const curClusterPaths: string[] = []
              setClusterPaths(curClusterPaths)
            }
          })
        }}><span className="glyphicon glyphicon-refresh">Cluster</span></button>}
  
      </div>
      <svg width={width} height={height} ref={svgRef} style={{ border: "1px dashed gray" }} onClick={() => {setHoveredItem(-1)}}>
        {clusterPaths.length > 0 && <g className="clusters">
          {clusterPaths.map((path, i) => (
            <path
              key={`cluster-${i}`}
              d={path}
              stroke='none'
              fill={colorScale(analysisResult.selectedClasses[i].toString())}
              fillOpacity={0.3}
            />
          ))}
        </g>}
        {showLines && <g className="lines">
          {hoveredItem !== -1 && distances[hoveredItem].map((dist, i) => (<>
            <line
              key={`line-${i}`}
              x1={xScale(x(coords[hoveredItem]))}
              y1={yScale(y(coords[hoveredItem]))}
              x2={xScale(x(coords[i]))}
              y2={yScale(y(coords[i]))}
              stroke='black'
              strokeWidth={opacityScale(dist) * 5}
              strokeOpacity={opacityScale(dist)}
            />
            <text
              key={`text-${i}`}
              x={xScale(x(coords[i])) + 10}
              y={yScale(y(coords[i])) + 10}
              fontSize={10}
              textAnchor='middle'
              alignmentBaseline='middle'
            >
              {dist.toFixed(2)}
            </text>
          </>))}
        </g>}
        <g className="points">
          {coords.map((point, i) => <circle
            key={`point-${i}`}
            cx={xScale(x(point))}
            cy={yScale(y(point))}
            // r={hoveredItem === -1 || hoveredItem === i ? 4 : opacityScale(distances[hoveredItem][i]) * 16 + 1}
            r={3}
            fill={analysisResult.selectedImages.includes(i) ? 'black' : colorScale(labels[i].toString())}
            // stroke={hoveredItem === i ? 'red' : (preds[i] ? (hoveredItem !== -1 ? 'lightgray' : 'none') : 'black')}
            stroke={hoveredItem === i ? 'red' : '#00000000'}
            // strokeWidth={hoveredItem === i || !preds[i] ? 2 : (hoveredItem === -1 ? 0 : 1)}
            strokeWidth={hoveredItem === -1 ? 0 : 1}
            data-tooltip-id="image-tooltip"
            onClick={(e) => {
              setHoveredItem(i)
              e.stopPropagation()
            }}
          />)}
        </g>
      </svg>
      { hoveredItem !== -1 && 
      <ImageToolTip
        imgs={[hoveredItem]}
        imgType={'raw'}
        imgData={{}}
        label={`Image ${hoveredItem}`}
      />}
    </div>
  )
}

export default ScatterPlot