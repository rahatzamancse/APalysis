import React from 'react';
import { useAppSelector } from '../app/hooks';
import { selectAnalysisResult } from '../features/analyzeSlice';
import * as d3 from 'd3';
import ImageToolTip from './ImageToolTip'

type Point = [number, number];
const x = (d: Point) => d[0];
const y = (d: Point) => d[1];

const svgMargin = 10

function ScatterPlot({ coords, preds, distances, labels, width, height }: {
  coords: Point[];
  preds: boolean[];
  distances: number[][];
  labels: number[];
  width: number;
  height: number;
}) {
  const analysisResult = useAppSelector(selectAnalysisResult)
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [hoveredItem, setHoveredItem] = React.useState<number>(-1)
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
  
  return (
    <div>
      <svg width={width} height={height} ref={svgRef} style={{ border: "1px dashed gray" }}>
        <g className="lines">
          {hoveredItem !== -1 && distances[hoveredItem].map((dist, i) => (<>
            <line
              key={`line-${i}`}
              x1={xScale(x(coords[hoveredItem]))}
              y1={yScale(y(coords[hoveredItem]))}
              x2={xScale(x(coords[i]))}
              y2={yScale(y(coords[i]))}
              stroke='black'
              strokeWidth={1}
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
        </g>
        <g className="points">
          {coords.map((point, i) => (
            <circle
              key={`point-${i}`}
              cx={xScale(x(point))}
              cy={yScale(y(point))}
              r={5}
              fill={analysisResult.selectedImages.includes(i) ? 'black' : colorScale(labels[i].toString())}
              data-tooltip-id="image-tooltip"
              onClick={() => {
                setHoveredItem(i)
              }}
              onMouseLeave={() => {
                setHoveredItem(-1)
              }}
              stroke={preds[i] ? 'none' : 'black'}
              strokeWidth={3}
            />
          ))}
        </g>
      </svg>
      <ImageToolTip
        imgs={[hoveredItem]}
        imgType={'raw'}
        imgData={{}}
      />
    </div>
  )
}

export default ScatterPlot