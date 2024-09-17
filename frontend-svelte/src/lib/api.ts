import type { ModelGraph } from "$lib/types";
import type { Node, Edge } from "$lib/types";

// const API_URL = "/api"
const API_URL = "http://localhost:8000/api"

export function getModelGraph(): Promise<ModelGraph> {

    return fetch(`${API_URL}/model/`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => ({
                nodes: data.graph.nodes.map((node: Node) => ({
                    id: node.id,
                    label: node.name,
                    layer_type: node.layer_type,
                    name: node.name,
                    tensor_type: node.tensor_type,
                    output_shape: node.is_parent ? undefined : node.output_shape,
                    is_parent: node.is_parent,
                    parent: node.parent,
                })),
                edges: data.graph.edges.map((edge: Edge) => ({
                    source: edge.source,
                    target: edge.target,
                }))
            })
        )
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

export function getActivationsImages(node: Node, startFilter: number, nFilters: number, nImgs: number): Promise<string[][]> {
    const imgLayerTypes = ["Conv2D", "MaxPooling2D", "AveragePooling2D", "Conv2d", "Cat", "Add", "Concatenate"]
    if(!imgLayerTypes.includes(node.layer_type)) {
        return Promise.resolve([])
    }
    
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

export function getTotalInputs(): Promise<number> {
    return fetch(`${API_URL}/total_inputs`)
        .then(response => response.json())
        .then(data => data['total'])
}

export function getAnalysisLayerCoords(node: string, method: string = 'mds', distance: string = 'euclidean', normalization: string = 'none', takeSummary: boolean = true): Promise<[number, number][]> {
    return fetch(`${API_URL}/analysis/layer/${node}/embedding?` + new URLSearchParams({
        normalization: normalization, // none, row, col
        method: method, // mds, tsne
        distance: distance, // jaccard, euclidean
        take_summary: takeSummary.toString(),
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