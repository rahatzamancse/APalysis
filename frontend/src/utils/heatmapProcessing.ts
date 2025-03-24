import { findIndicesOfMax, transposeArray } from '@utils/utils'

export interface JaccardResult {
    intersection: number
    union: number
    similarity: number
}

export function normalizeHeatmap(heatmap: number[][]): number[][] {
    return transposeArray(transposeArray(heatmap).map(row => {
        // Mean shift
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const meanShiftedRow = row.map(item => item - mean)
        
        // Normalize
        const max = Math.max(...meanShiftedRow)
        const min = Math.min(...meanShiftedRow)
        const normalRow = meanShiftedRow.map(item => (item - min) / (max - min))
        return normalRow
    }))
}

export function filterTopChannels(heatmap: number[][]): number[][] {
    const TOTAL_MAX_CHANNELS = (arr: number[]) => arr.length * 0.5
    const indicesMax = heatmap.map(arr => findIndicesOfMax(arr, TOTAL_MAX_CHANNELS(arr)))
    
    return heatmap.map((col, i) => {
        const newCol = [...col]
        col.forEach((_, j) => {
            if(!indicesMax[i].includes(j)) {
                newCol[j] = 0
            }
        })
        return newCol
    })
}

export function calculateJaccardSimilarity(col1: number[], col2: number[]): JaccardResult {
    const intersection = col1
        .map((item, k) => (item > 0 && col2[k] > 0))
        .reduce((total, x) => total + (x?1:0), 0)
    const union = col1
        .map((item, k) => (item > 0 || col2[k] > 0))
        .reduce((total, x) => total + (x?1:0), 0)
    return {
        intersection,
        union,
        similarity: intersection / union
    }
}

export function calculatePairwiseJaccard(heatmap: number[][]): JaccardResult[][] {
    return heatmap.map((col1, i) => {
        return heatmap.map((col2, j) => {
            if (i === j) return { intersection: 1, union: 1, similarity: 1 }
            return calculateJaccardSimilarity(col1, col2)
        })
    })
} 