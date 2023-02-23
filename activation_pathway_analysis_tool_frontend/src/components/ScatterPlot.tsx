import React from 'react';
import * as api from '../api'
import { useAppDispatch } from '../app/hooks';
import { useAppSelector } from '../app/hooks';
import { setSelectedImgs, selectAnalysisResult } from '../features/analyzeSlice';
import * as d3 from 'd3';

type Point = [number, number];
const x = (d: Point) => d[0];
const y = (d: Point) => d[1];

const tooltipWidth = 60
const tooltipHeight = 60

type Props = {
  coords: Point[];
  labels: number[];
  width: number;
  height: number;
};

let tooltipTimeout: number
const svgMargin = 10

function ScatterPlot({ coords, labels, width, height }: Props) {
  const dispatch = useAppDispatch()
  const analysisResult = useAppSelector(selectAnalysisResult)
  const [tooltipImg, setTooltipImg] = React.useState('')
  const svgRef = React.useRef<SVGSVGElement>(null);
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

  // event handlers
  // const handlePointHover = useCallback(
  //   (event: React.MouseEvent | React.TouchEvent) => {
  //     if (!svgRef.current) return;
  //     const point = [1,2] as Point

  //     const idx = coords.indexOf(point)
  //     api.getAnalysisImage(idx).then((res) => {
  //       setTooltipImg(res)
  //     })
  //     showTooltip({
  //       tooltipLeft: xScale(x(point)),
  //       tooltipTop: yScale(y(point)),
  //       tooltipData: point,
  //     });
  //   },
  //   [xScale, yScale, showTooltip],
  // );

  // const handleMouseLeave = useCallback(() => {
  //   tooltipTimeout = window.setTimeout(() => {
  //     hideTooltip();
  //   }, 300);
  // }, [hideTooltip]);

  // const onBrushChange = (event) => {
  //   const { x0, x1, y0, y1 } = domain;

  //   // Get the points within the brush
  //   const selectedPoints = coords.filter((point) => {
  //     const xVal = xScale(x(point));
  //     const yVal = yScale(y(point));

  //     return xVal >= x0 && xVal <= x1 && yVal >= y0 && yVal <= y1;
  //   });
  //   const selectedLabels = selectedPoints.map((point) => labels[coords.findIndex(p => x(p) === x(point) && y(p) === y(point))])
  //   dispatch(setSelectedImgs(selectedLabels))
  // };
  // 

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
              fill={analysisResult.selectedImgs.includes(i) ? 'black' : colorScale(labels[i].toString())}
            />
          ))}
        </g>
      </svg>
      {/* {tooltipOpen && tooltipData && tooltipLeft != null && tooltipTop != null && (
          <Tooltip left={tooltipLeft-tooltipWidth/2} top={tooltipTop-tooltipHeight/2}>
            <img src={tooltipImg} width={tooltipWidth} height={tooltipHeight} />
          </Tooltip>
        )} */}
    </div>
  )
}

export default ScatterPlot