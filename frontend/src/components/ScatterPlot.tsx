import React from 'react';
import { useAppSelector } from '../app/hooks';
import { selectAnalysisResult } from '../features/analyzeSlice';
import * as d3 from 'd3';
import ImageToolTip from './ImageToolTip'
import { Node } from '../types';
import * as api from '../api';
import smoothHull from '../convexHull';
import { polygonHull } from 'd3';

type Point = [number, number];
const x = (d: Point) => d[0];
const y = (d: Point) => d[1];

const svgMargin = 10

const LegendItem = ({ color, counts, scale, unitBarWidth=10, height=20}: {
  color: string,
  counts: { [key: number]: number },
  scale: d3.ScaleOrdinal<string, string>,
  unitBarWidth: number,
  height?: number
}) => {
  const [hoveredKey, setHoveredKey] = React.useState<number | null>(null);
  
  console.log("unitBarWidth", unitBarWidth)

  return <g>
      <rect
        x={0}
        y={0}
        width={30}
        height={20}
        rx={5}
        ry={5}
        fill={color}
      />
      <circle cx={8} cy={7} r={2} fill="gray" />
      <circle cx={15} cy={13} r={2} fill="gray" />
      <circle cx={22} cy={10} r={2} fill="gray" />
      
      {Object.entries(counts).map(([key, value], i) => {
        const prevValues = Object.values(counts)
          .slice(0, i)
          .reduce((sum, v) => sum + v, 0);
        const width = value * unitBarWidth;
        const xPos = prevValues * unitBarWidth + 30 + 10;
        return (
          <g key={key}>
            <rect
              x={xPos}
              y={0} 
              width={width}
              height={height}
              fill={scale(key)}
              onMouseEnter={() => setHoveredKey(Number(key))}
              onMouseLeave={() => setHoveredKey(null)}
            />
            {hoveredKey === Number(key) && (
              <text
                x={xPos + width/2}
                y={height/2+5}
                textAnchor="middle"
                fontSize={12}
              >
                {value}
              </text>
            )}
          </g>
        );
      })}
    </g>
};



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
  const [useXMeans, setUseXMeans] = React.useState<boolean>(true)
  const [kClusters, setKClusters] = React.useState<number>(2)
  const analysisResult = useAppSelector(selectAnalysisResult)
  const [hoveredItem, setHoveredItem] = React.useState<number>(-1)
  const [clusterPaths, setClusterPaths] = React.useState<string[]>([])
  const [showImages, setShowImages] = React.useState<boolean>(false)
  const [classes, setClasses] = React.useState<string[]>([])
  const [clusterMembersCount, setClusterMembersCount] = React.useState<{[key: string]: number}[]>([])
  
  console.log("clusterMembersCount", clusterMembersCount)

  React.useEffect(() => {
    api.getLabels().then(setClasses)
  }, [])

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
  
  const clusterColorScale = React.useMemo(
    () => d3.scaleOrdinal<string>()
      .domain(Array.from({length: clusterPaths.length}, (_, i) => i.toString()))
      .range(d3.schemePastel1.slice()),
    [clusterPaths]
  )
  
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

  const imageSize = 30
  
  return (
    <div style={{ display: 'flex', gap: '20px' }}>
      <div>
        <svg width={width} height={height} style={{ border: "1px dashed gray" }} onClick={() => {setHoveredItem(-1)}}>
          {clusterPaths.length > 0 && <g className="clusters">
            {clusterPaths.map((path, i) => (
              <path
                key={`cluster-${i}`}
                d={path}
                stroke='none'
                fill={clusterColorScale(i.toString())}
                fillOpacity={0.3}
              />
            ))}
          </g>}
          {showLines && <g className="lines">
            {hoveredItem !== -1 && distances[hoveredItem].map((dist, i) => (<g key={`line-${i}`}>
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
            </g>))}
          </g>}
          <g className="points">
            {coords.map((point, i) => (
              showImages ? (
                <g>
                  <rect
                    key={`border-${i}`}
                    x={xScale(x(point)) - imageSize/2}
                    y={yScale(y(point)) - imageSize/2}
                    width={imageSize}
                    height={imageSize}
                    fill="none"
                    stroke={analysisResult.selectedImages.includes(i) ? 'black' : colorScale(labels[i].toString())}
                    strokeWidth={3}
                  />
                  <image
                    key={`point-${i}`}
                    href={api.getInputImageURL(i)}
                    x={xScale(x(point)) - imageSize/2}
                    y={yScale(y(point)) - imageSize/2}
                    width={imageSize}
                    height={imageSize}
                    data-tooltip-id="image-tooltip"
                    onClick={(e) => {
                      setHoveredItem(i)
                      e.stopPropagation()
                    }}
                  />
                </g>
              ) : (
                <circle
                  key={`point-${i}`}
                  cx={xScale(x(point))}
                  cy={yScale(y(point))}
                  r={3}
                  fill={analysisResult.selectedImages.includes(i) ? 'black' : colorScale(labels[i].toString())}
                  stroke={hoveredItem === i ? 'red' : '#00000000'}
                  strokeWidth={hoveredItem === -1 ? 0 : 1}
                  data-tooltip-id="image-tooltip"
                  onClick={(e) => {
                    setHoveredItem(i)
                    e.stopPropagation()
                  }}
                />
              )
            ))}
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
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {node && (
          <>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button 
                type="button" 
                className={`btn ${showImages ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setShowImages(!showImages)}
              >
                {showImages ? 'Show Points' : 'Show Images'}
              </button>
            </div>

            <label style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <input
                type="checkbox"
                checked={useXMeans}
                onChange={(e) => setUseXMeans(e.target.checked)}
              /> 
              Use X-Means
            </label>
            
            {!useXMeans && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <label>Number of clusters:</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={kClusters}
                  onChange={(e) => setKClusters(Math.min(10, Math.max(1, parseInt(e.target.value) || 1)))}
                  style={{ width: '60px' }}
                />
              </div>
            )}
            
            <button 
              type="button" 
              className="btn btn-primary"
              onClick={(e) => {
                api.getCluster(node.name, useXMeans, kClusters).then((clusters) => {
                  if(clusters.centers.length > 0) {
                    const curClusterPaths: string[] = []
                    clusters.centers.forEach((c, i) => {
                      const clusterIndices = clusters.labels.map((l, j) => i === l ? j : -1).filter((x) => x !== -1)
                      let hullPoints = polygonHull(coords.filter((d, i) => clusterIndices.includes(i) && !clusters.outliers.includes(i)).map(d => [xScale(x(d)), yScale(y(d))]))
                      const maxHullPadding = 20
                      const minHullPadding = 10
                      const hullPadding = Math.floor(Math.random() * (maxHullPadding - minHullPadding + 1) + minHullPadding)
                      if (hullPoints === null) {
                        hullPoints = coords.filter((d, i) => clusterIndices.includes(i) && !clusters.outliers.includes(i)).map(d => [xScale(x(d)), yScale(y(d))])
                      }
                      const hullPath = smoothHull(hullPoints, hullPadding)
                      curClusterPaths.push(hullPath)
                    })

                    const curClusterMembersCount: {[key: string]: number}[] = Array.from({length: curClusterPaths.length}, () => ({}))
                    clusters.labels.forEach((c, i) => {
                      const classIndex = analysisResult.selectedClasses[(Math.floor(i / analysisResult.examplePerClass))]
                      if (curClusterMembersCount[c][classIndex] === undefined)
                        curClusterMembersCount[c][classIndex] = 0
                      curClusterMembersCount[c][classIndex] += 1
                    })
                    setClusterMembersCount(curClusterMembersCount)
                    setClusterPaths(curClusterPaths)
                  }
                })
              }}
            >
              <span className="glyphicon glyphicon-refresh">Cluster</span>
            </button>

            <div style={{ marginTop: '20px', display: 'flex', gap: '20px' }}>
              <div>
                <h5>Inputs</h5>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {Array.from(new Set(labels)).map((label, i) => (
                    <div key={`legend-${label}`} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <svg width="40" height="20">
                        {!showImages ? (
                          <circle
                            cx="20"
                            cy="10"
                            r="3"
                            fill={colorScale(label.toString())}
                          />
                        ) : (
                          <>
                            <rect
                              x="15"
                              y="5"
                              width="10"
                              height="10"
                              fill={colorScale(label.toString())}
                              stroke={colorScale(label.toString())}
                              strokeWidth="1"
                            />
                            <text
                              x="20"
                              y="13"
                              textAnchor="middle"
                              fontSize="8"
                              fill="white"
                            >
                              📷
                            </text>
                          </>
                        )}
                      </svg>
                      <span>{classes[label]}</span>
                    </div>
                  ))}
                  {analysisResult.selectedImages.length > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <svg width="40" height="20">
                        <circle
                          cx="20"
                          cy="10"
                          r="3"
                          fill="black"
                        />
                      </svg>
                      <span>Selected Images</span>
                    </div>
                  )}
                </div>
              </div>
              <div>
                <h5>Clusters</h5>
                <svg width="200" height={clusterPaths.length * 30}>
                  {clusterPaths.filter(clusterPath => clusterPath !== "").map((_, i) => (
                    <g transform={`translate(0, ${i * 30})`} key={`cluster-${i}`}>
                      <LegendItem
                        color={clusterColorScale(i.toString())}
                        counts={clusterMembersCount[i]}
                        scale={colorScale}
                        unitBarWidth={120/Math.max(...clusterMembersCount.map(cCount => Object.values(cCount).reduce((a,b) => a+b, 0)))}
                      />
                    </g>
                  ))}
                </svg>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default ScatterPlot