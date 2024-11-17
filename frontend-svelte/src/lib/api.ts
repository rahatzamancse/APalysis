import type { ModelGraph } from "$lib/types";
import type { FunctionNode, TensorNode, ContainerNode, LayerEdge } from "$lib/types";

// const API_URL = "/api"
const API_URL = "http://localhost:8000/api"

export function getModelGraph(): Promise<ModelGraph> {
    return fetch(`${API_URL}/model/`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => {
            const nodes: (FunctionNode | TensorNode | ContainerNode)[] = []
            data.graph.nodes.forEach((node: any) => {
                if (node.type === 'function') {
                    nodes.push({
                        id: node.id,
                        name: node.name,
                        input_shape: node.input_shape,
                        output_shape: node.output_shape,
                        node_type: node.type,
                    } as FunctionNode)
                } else if (node.type === 'tensor') {
                    nodes.push({
                        id: node.id,
                        name: node.name,
                        shape: node.shape,
                        node_type: node.type,
                    } as TensorNode)
                } else if (node.type === 'container') {
                    nodes.push({
                        id: node.id,
                        name: node.name,
                        children: node.children ?? [],
                        node_type: node.type,
                    } as ContainerNode)
                }
            })
            const edges: LayerEdge[] = data.graph.edges.map((edge: any) => ({
                source: edge.source,
                target: edge.target,
                label: edge.label,
                type: 'smoothstep',
            } as LayerEdge)) 
            return { nodes, edges }
        })
}

export function expandNode(node: string): Promise<ModelGraph> {
    return fetch(`${API_URL}/model/expand`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node: node }),
    })
        .then(response => response.json())
        .then(data => data)
}

export function collapseNode(node: string): Promise<ModelGraph> {
    return fetch(`${API_URL}/model/collapse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node: node }),
    })
        .then(response => response.json())
        .then(data => data)
}

export function saveDataset(): Promise<string> {
    return fetch(`${API_URL}/analysis/images/save`, {
        method: "POST",
    })
        .then(response => response.status === 200 ? response.blob() : Promise.reject(response))
        .then(blob => {
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.style.display = 'none'
            a.href = url
            a.download = 'dataset.zip'
            a.click()
            return url
        })
        .catch(err => {
            console.error(err)
            return ""
        })
}

export function getCluster(layer: string): Promise<{ labels: number[], centers: number[][], distances: number[], outliers: number[]}> {
    return fetch(`${API_URL}/analysis/layer/${layer}/cluster?` + new URLSearchParams({
        outlier_threshold: '2',
    }))
        .then(response => response.json())
        .then(data => data)
}

export function getActivationsImages(node: TensorNode, startFilter: number, nFilters: number, nImgs: number): Promise<string[][]> {
    const promises: Promise<string>[][] = []
    Array.from(Array(nFilters).keys(), x => x + startFilter).forEach(filterIdx => 
        promises.push(
            Array.from(Array(nImgs).keys()).map(imgIdx =>
                fetch(`${API_URL}/analysis/image/${imgIdx}/layer/${node.name}/filter/${filterIdx}`)
                .then(response => response.blob())
                .then(blob => URL.createObjectURL(blob))
            )
        )
    )
    
    return Promise.all(promises.map(p => Promise.all(p)))
}

export function getInputShape(index: number): Promise<number[]> {
    return fetch(`${API_URL}/input/${index}/shape`)
        .then(response => response.json())
        .then(data => data)
}

export function getInputImage(index: number): Promise<string> {
    return fetch(`${API_URL}/input/${index}`)
        .then(response => response.blob())
        .then(blob => URL.createObjectURL(blob))
}

export function getTotalInputs(): Promise<number> {
    return fetch(`${API_URL}/total_inputs`)
        .then(response => response.json())
        .then(data => data['total'])
}

export function getProjection(tensorId: string, method: 'mds' | 'tsne' | 'umap' | 'pca' = 'pca', distance: 'jaccard' | 'euclidean' = 'euclidean', normalization: 'none' | 'row' | 'col' = 'none'): Promise<[number, number][]> {
    return fetch(`${API_URL}/analysis/${tensorId}/projection?` + new URLSearchParams({
        normalization: normalization,
        method: method,
        distance: distance,
    }))
        .then(response => response.json())
        .then(data => data)
}

export function getAnalysisDistanceMatrix(node: string): Promise<number[][]> {
    return fetch(`${API_URL}/analysis/layer/${node}/embedding/distance`)
        .then(response => response.json())
        .then(data => data)
}

export function getAllDistances(): Promise<number[][]> {
    return fetch(`${API_URL}/analysis/alldistances`)
        .then(response => response.json())
        .then(data => data)
}

export function getAnalysisHeatmap(node: string): Promise<number[][]> {
    return fetch(`${API_URL}/analysis/layer/${node}/heatmap`)
        .then(response => response.json())
        .then(data => data)
}

export function getTaskStatus(task_id: string): Promise<{ message: string, task_id: string }> {
    return fetch(`${API_URL}/taskStatus?task_id=${task_id}`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
    .then(response => response.json())
    .then(data => data)
}

export function getDenseArgmax(layer: string): Promise<number[]> {
    return fetch(`${API_URL}/analysis/layer/${layer}/argmax`)
        .then(response => response.json())
        .then(data => data)
}


export function getKernel(node: string, channel: number): Promise<string> {
    return fetch(`${API_URL}/analysis/layer/${node}/${channel}/kernel`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.blob())
        .then(blob => URL.createObjectURL(blob))
}

export function getAllEmbedding(): Promise<[number, number][]> {
    return fetch(`${API_URL}/analysis/allembedding`)
        .then(response => response.json())
        .then(data => data)
}