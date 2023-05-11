import React from 'react';
import * as api from '../api'
import { useAppSelector } from '../app/hooks';
import { selectAnalysisResult } from '../features/analyzeSlice';
import * as d3 from 'd3';
import ImageToolTip from './ImageToolTip'

type Point = [number, number];
const x = (d: Point) => d[0];
const y = (d: Point) => d[1];

const svgMargin = 10

function ScatterPlot({ coords, preds, labels, width, height }: {
  coords: Point[];
  labels: number[];
  width: number;
  height: number;
  preds: boolean[];
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

  return (
    <div>
      <svg width={width} height={height} ref={svgRef} style={{ border: "1px dashed gray" }}>
        <g>
          {coords.map((point, i) => (
            <circle
              key={`point-${i}`}
              cx={xScale(x(point))}
              cy={yScale(y(point))}
              r={5}
              fill={analysisResult.selectedImages.includes(i) ? 'black' : colorScale(labels[i].toString())}
              data-tooltip-id="image-tooltip"
              onMouseEnter={() => {
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