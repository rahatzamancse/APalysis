import { ModelGraph } from "./types";
import { Node } from "./types";

// const API_URL = "/api"
const API_URL = "http://localhost:8000/api"

export function getModelGraph(): Promise<ModelGraph> {

    return fetch(`${API_URL}/model/`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        // body: JSON.stringify({})
    })
        .then(response => response.json())
        .then(data => ({
            nodes: data.graph.nodes.map((node: any) => ({
                id: node.id,
                label: node.label,
                layer_type: node.layer_type,
                name: node.name,
                input_shape: node.input_shape,
                kernel_size: node.kernel_size,
                output_shape: node.output_shape,
                tensor_type: node.tensor_type,
                pos: node.pos? { x: node.pos.x, y: node.pos.y } : undefined,
                out_edge_weight: data.edge_weights[node.name],
            })),
            edges: data.graph.links.map((edge: any) => ({
                source: edge.source,
                target: edge.target,
            })),
            meta: {
                depth: data.meta.depth,
            },
        }))
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

export function getTaskStatus(task_id: string): Promise<{ message: string, task_id: string, payload: any }> {
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