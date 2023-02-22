import React, { useMemo, useCallback, useRef } from 'react';
import { Group } from '@visx/group';
import { Circle } from '@visx/shape';
import { scaleLinear } from '@visx/scale';
import { withTooltip, Tooltip } from '@visx/tooltip';
import { WithTooltipProvidedProps } from '@visx/tooltip/lib/enhancers/withTooltip';
import { localPoint } from '@visx/event';
import { voronoi } from '@visx/voronoi';
import { schemeCategory10 } from 'd3-scale-chromatic';
import { scaleOrdinal } from '@visx/scale';
import { Brush } from '@visx/brush';
import BaseBrush, { BaseBrushState, UpdateBrush } from '@visx/brush/lib/BaseBrush';
import { Bounds } from '@visx/brush/lib/types';
import * as api from '../api'

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
  showControls?: boolean;
};

let tooltipTimeout: number

export default withTooltip<Props, Point>(
  ({
    coords,
    labels,
    width,
    height,
    showControls = true,
    hideTooltip,
    showTooltip,
    tooltipOpen,
    tooltipData,
    tooltipLeft,
    tooltipTop,
  }: Props & WithTooltipProvidedProps<Point>) => {
    if (width < 10) return null;
    const [tooltipImg, setTooltipImg] = React.useState('')
    const svgRef = useRef<SVGSVGElement>(null);
    const xScale = useMemo(
      () =>
        scaleLinear<number>({
          domain: [Math.min(...coords.map(x)), Math.max(...coords.map(x))],
          range: [0, width],
          clamp: true,
        }),
      [width],
    );
    const yScale = useMemo(
      () =>
        scaleLinear<number>({
          domain: [Math.min(...coords.map(y)), Math.max(...coords.map(y))],
          range: [height, 0],
          clamp: true,
        }),
      [height],
    );
    const colorScale = scaleOrdinal({
      domain: Array.from(new Set(labels)),
      range: schemeCategory10.slice(),
    });
    const voronoiLayout = useMemo(
      () =>
        voronoi<Point>({
          x: (d) => xScale(x(d)) ?? 0,
          y: (d) => yScale(y(d)) ?? 0,
          width,
          height,
        })(coords),
      [width, height, xScale, yScale],
    );

    // event handlers
    const handleMouseMove = useCallback(
      (event: React.MouseEvent | React.TouchEvent) => {
        if (!svgRef.current) return;

        // find the nearest polygon to the current mouse position
        const point = localPoint(svgRef.current, event);
        if (!point) return;
        const neighborRadius = 100;
        const closest = voronoiLayout.find(point.x, point.y, neighborRadius);
        if (closest) {
          const idx = coords.indexOf(closest.data)
          api.getAnalysisImage(idx).then((res) => {
            setTooltipImg(res)
          })
          showTooltip({
            tooltipLeft: xScale(x(closest.data)),
            tooltipTop: yScale(y(closest.data)),
            tooltipData: closest.data,
          });
        }
      },
      [xScale, yScale, showTooltip, voronoiLayout],
    );

    const handleMouseLeave = useCallback(() => {
      tooltipTimeout = window.setTimeout(() => {
        hideTooltip();
      }, 300);
    }, [hideTooltip]);

    const brushRef = useRef<BaseBrush | null>(null);

    const onBrushChange = (domain: Bounds | null) => {
      if (!domain) return;
      const { x0, x1, y0, y1 } = domain;
      console.log(x0, x1, y0, y1);
    };


    return (
      <div>
        <svg width={width} height={height} ref={svgRef}>
          {/** capture all mouse events with a rect */}
          <rect
            width={width}
            height={height}
            rx={14}
            fill="#00000000"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            onTouchMove={handleMouseMove}
            onTouchEnd={handleMouseLeave}
          />
          <Group pointerEvents="none">
            {coords.map((point, i) => (
              <Circle
                key={`point-${i}`}
                cx={xScale(x(point))}
                cy={yScale(y(point))}
                r={5}
                fill={tooltipData === point ? 'black' : colorScale(labels[i])}
              />
            ))}
            <Brush
              xScale={xScale}
              yScale={yScale}
              width={width}
              height={height}
              margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
              handleSize={8}
              innerRef={brushRef}
              resizeTriggerAreas={['left', 'right', 'top', 'bottom']}
              brushDirection="both"
              initialBrushPosition={{ start: { x: 0, y: height/2}, end: {x: width/2, y: 0}}}
              onChange={onBrushChange}
              // onClick={() => setFilteredStock(stock)}
              selectedBoxStyle={{
                fill: "#3b697844",
                stroke: 'white',
              }}
              useWindowMoveEvents
              // renderBrushHandle={(props) => <BrushHandle {...props} />}
            />
          </Group>
        </svg>
        {tooltipOpen && tooltipData && tooltipLeft != null && tooltipTop != null && (
          <Tooltip left={tooltipLeft-tooltipWidth/2} top={tooltipTop-tooltipHeight/2}>
            <img src={tooltipImg} width={tooltipWidth} height={tooltipHeight} />
          </Tooltip>
        )}
      </div>
    );
  },
);