import React from 'react';
import { Node } from '../types';
import { useAppSelector } from '../app/hooks';
import { selectAnalysisResult } from '../features/analyzeSlice';
import * as api from '../api';
import { transposeArray, findIndicesOfMax } from '../utils';

interface HierarchyTreeProps {
    node: Node;
}

function HierarchyTree({ node }: HierarchyTreeProps) {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const analyzeResult = useAppSelector(selectAnalysisResult)
    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        api.getAnalysisHeatmap(node.name).then(setHeatmap)
    }, [node.name, analyzeResult])

    if (heatmap.length === 0) return null
    const nExamples = analyzeResult.examplePerClass * analyzeResult.selectedClasses.length
        
    // Normalize all rows in heatmap
    const finalHeatmap = transposeArray(transposeArray(heatmap).map(row => {
        // Mean shift
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const meanShiftedRow = row.map(item => item - mean)
        
        // Normalize
        const max = Math.max(...meanShiftedRow)
        const min = Math.min(...meanShiftedRow)
        const normalRow = meanShiftedRow.map(item => (item - min) / (max - min))
        return normalRow
    }))

    // Choose only first TOTAL_MAX_CHANNELS elements of each row
    const TOTAL_MAX_CHANNELS = (arr: number[]) => arr.length * 0.5
    // const TOTAL_MAX_CHANNELS = (arr: number[]) => 10
    const indicesMax = finalHeatmap.map(arr => findIndicesOfMax(arr, TOTAL_MAX_CHANNELS(arr)))
    finalHeatmap.forEach((col, i) => {
        // Make all other elements of arr 0 except the max elements
        col.forEach((_, j) => {
            if(!indicesMax[i].includes(j)){
                col[j] = 0
            }
        })
    })
    
    // Calculate Jaccard similarity between each pair of columns
    const jDist = finalHeatmap.map((col1, i) => {
        return finalHeatmap.map((col2, j) => {
            if (i === j) return [1, 1, 1]
            const intersection = col1
                .map((item, k) => (item > 0 && col2[k] > 0))
                .reduce((total, x) => total + (x?1:0), 0)
            const union = col1
                .map((item, k) => (item > 0 || col2[k] > 0))
                .reduce((total, x) => total + (x?1:0), 0)
            return intersection / union
        })
    })

    // jDist is a 2D array of size nExamples x nExamples of similarity
    // Calculate mean pairwise distances between classes
    const nClasses = analyzeResult.selectedClasses.length
    const classDistances = Array(nClasses).fill(0).map(() => Array(nClasses).fill(0))
    
    for(let i = 0; i < nClasses; i++) {
        for(let j = 0; j < nClasses; j++) {
            let sum = 0
            let count = 0
            // For each pair of examples in classes i and j
            for(let k = i*analyzeResult.examplePerClass; k < (i+1)*analyzeResult.examplePerClass; k++) {
                for(let l = j*analyzeResult.examplePerClass; l < (j+1)*analyzeResult.examplePerClass; l++) {
                    if(k !== l) { // Skip comparing same examples
                        sum += jDist[k][l]
                        count++
                    }
                }
            }
            classDistances[i][j] = sum / count
        }
    }

    
    console.log(jDist)
    
    
    return (
        <div>
            {/* Tree visualization will go here */}
        </div>
    );
}

export default HierarchyTree;
