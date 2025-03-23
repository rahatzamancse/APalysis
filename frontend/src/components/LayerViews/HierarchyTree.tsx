import React from 'react';
import { Node } from '@types';
import { useAppSelector } from '@hooks';
import { selectAnalysisResult } from '@features/analyzeSlice';
import * as api from '@api';
import { transposeArray, findIndicesOfMax } from '@utils';
import * as d3 from 'd3';

interface HierarchyTreeProps {
    node: Node;
}

// Move ClusterNode type definition outside
type ClusterNode = {
    id: number,
    children: ClusterNode[],
    isLeaf: boolean,
    similarity?: number
}

function HierarchyTree({ node }: HierarchyTreeProps) {
    // 1. All hooks at the top
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const [classLabels, setClassLabels] = React.useState<string[]>([])
    const [tree, setTree] = React.useState<ClusterNode | null>(null)
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const svgRef = React.useRef<SVGSVGElement>(null)
    const [width, height] = [600, 400]

    // 2. Data fetching effect
    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        api.getAnalysisHeatmap(node.name).then(setHeatmap)
        api.getLabels().then(setClassLabels)
    }, [node.name, analyzeResult])

    // 3. Tree computation effect
    React.useEffect(() => {
        if (heatmap.length === 0) return

        const nExamples = analyzeResult.examplePerClass * analyzeResult.selectedClasses.length
        
        // Normalize all rows in heatmap
        const finalHeatmap = transposeArray(transposeArray(heatmap).map(row => {
            const mean = row.reduce((a, b) => a + b, 0) / row.length
            const meanShiftedRow = row.map(item => item - mean)
            const max = Math.max(...meanShiftedRow)
            const min = Math.min(...meanShiftedRow)
            const normalRow = meanShiftedRow.map(item => (item - min) / (max - min))
            return normalRow
        }))

        // Choose only first TOTAL_MAX_CHANNELS elements of each row
        const TOTAL_MAX_CHANNELS = (arr: number[]) => arr.length * 0.5
        const indicesMax = finalHeatmap.map(arr => findIndicesOfMax(arr, TOTAL_MAX_CHANNELS(arr)))
        finalHeatmap.forEach((col, i) => {
            col.forEach((_, j) => {
                if(!indicesMax[i].includes(j)) col[j] = 0
            })
        })
        
        // Calculate Jaccard similarity
        const jDist = finalHeatmap.map((col1, i) => {
            return finalHeatmap.map((col2, j) => {
                if (i === j) return 1
                const intersection = col1
                    .map((item, k) => (item > 0 && col2[k] > 0))
                    .reduce((total, x) => total + (x?1:0), 0)
                const union = col1
                    .map((item, k) => (item > 0 || col2[k] > 0))
                    .reduce((total, x) => total + (x?1:0), 0)
                return intersection / union
            })
        })

        // Calculate class similarities
        const nClasses = analyzeResult.selectedClasses.length
        const classSimilarities = Array(nClasses).fill(0).map<number[]>(() => Array(nClasses).fill(0))
        
        for(let i = 0; i < nClasses; i++) {
            for(let j = 0; j < nClasses; j++) {
                let sum = 0, count = 0
                for(let k = i*analyzeResult.examplePerClass; k < (i+1)*analyzeResult.examplePerClass; k++) {
                    for(let l = j*analyzeResult.examplePerClass; l < (j+1)*analyzeResult.examplePerClass; l++) {
                        if(k !== l) {
                            sum += jDist[k][l]
                            count++
                        }
                    }
                }
                classSimilarities[i][j] = sum / count
            }
        }
        
        console.log("Class similarities: ", classSimilarities)
        
        // Build cluster tree
        const buildClusterTree = () => {
            console.log("Step 1: Creating initial leaf nodes")
            let clusters: ClusterNode[] = analyzeResult.selectedClasses.map((_, idx) => ({
                id: idx,
                children: [],
                isLeaf: true
            }))
            console.log("Initial clusters:", clusters)
            
            let currentSimilarities = classSimilarities.map(row => [...row])
            let nextNodeId = clusters.length
            console.log("Initial similarities matrix:", currentSimilarities)

            while (clusters.length > 1) {
                console.log("\nStep 2: Finding most similar pair")
                let maxSim = -Infinity
                let toMerge = [-1, -1]
                
                for (let i = 0; i < currentSimilarities.length; i++) {
                    for (let j = i + 1; j < currentSimilarities.length; j++) {
                        if (currentSimilarities[i][j] > maxSim) {
                            maxSim = currentSimilarities[i][j]
                            toMerge = [i, j]
                        }
                    }
                }
                console.log("Most similar pair:", toMerge, "with similarity:", maxSim)

                const [i, j] = toMerge
                console.log("Step 3: Creating new merged node")
                const newNode = {
                    id: nextNodeId++,
                    children: [clusters[i], clusters[j]],
                    isLeaf: false,
                    similarity: maxSim
                }
                console.log("New node:", newNode)

                console.log("Step 4: Calculating new distances")
                const newDistances: number[] = []
                for (let k = 0; k < currentSimilarities.length; k++) {
                    if (k !== i && k !== j) {
                        newDistances.push((currentSimilarities[i][k] + currentSimilarities[j][k]) / 2)
                    }
                }
                console.log("New distances:", newDistances)

                console.log("Step 5: Updating similarity matrix")
                currentSimilarities = currentSimilarities
                    .filter((_, idx) => idx !== i && idx !== j)
                    .map(row => row.filter((_, idx) => idx !== i && idx !== j))
                
                currentSimilarities.push(newDistances)
                currentSimilarities.forEach((row, idx) => {
                    if (idx < currentSimilarities.length - 1) {
                        row.push(newDistances[idx])
                    }
                })
                console.log("Updated similarities matrix:", currentSimilarities)

                console.log("Step 6: Updating clusters list")
                clusters = clusters
                    .filter((_, idx) => idx !== i && idx !== j)
                    .concat([newNode])
                console.log("Updated clusters:", clusters)
            }

            console.log("Final tree:", clusters[0])
            return clusters[0]
        }

        setTree(buildClusterTree())
    }, [heatmap, analyzeResult])

    // 4. Visualization effect
    React.useEffect(() => {
        if (!svgRef.current || !tree) return

        d3.select(svgRef.current).selectAll("*").remove()

        const layout = d3.tree<ClusterNode>()
            .size([width - 40, height - 40])

        const root = d3.hierarchy(tree)
        const treeData = layout(root)

        const svg = d3.select(svgRef.current)
            .append("g")
            .attr("transform", `translate(20,20)`)

        svg.selectAll("path.link")
            .data(treeData.links())
            .enter()
            .append("path")
            .attr("class", "link")
            .attr("fill", "none")
            .attr("stroke", "#ccc")
            .attr("d", d => d3.linkVertical()({
                source: [d.source.x!, d.source.y!],
                target: [d.target.x!, d.target.y!]
            }))

        const nodes = svg.selectAll("g.node")
            .data(treeData.descendants())
            .enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${d.x},${d.y})`)

        nodes.append("circle")
            .attr("r", 5)
            .attr("fill", d => (d.data as ClusterNode).isLeaf ? "#4CAF50" : "#2196F3")

        nodes.append("text")
            .attr("dy", "-0.7em")
            .attr("x", 0)
            .attr("text-anchor", "middle")
            .text(d => {
                const nodeData = d.data as ClusterNode
                if (nodeData.isLeaf) {
                    const classId = analyzeResult.selectedClasses[nodeData.id]
                    return classLabels[classId] || classId.toString()
                }
                return `Cluster ${nodeData.id}`
            })
            .style("font-size", "12px")

    }, [tree, classLabels, analyzeResult.selectedClasses, width, height])

    if (heatmap.length === 0) return null

    return (
        <div>
            <svg ref={svgRef} width={width} height={height}></svg>
        </div>
    )
}

export default HierarchyTree;
