import { AnalysisConfig } from "./features/analyzeSlice";
import { ModelGraph } from "./types";
import { Node } from "./types";

const API_URL = "/api"
// const API_URL = "http://localhost:8000/api"

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

export function getFeatureActivatedChannels(): Promise<{activated_channels: {[layer: string]: number[]}}> {
    return fetch(`${API_URL}/polygon/activated_channels`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => data)
}

export function getLabels(): Promise<string[]> {
    return fetch(`${API_URL}/labels/`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => data)
}

export function getFeatureHuntImage(): Promise<string> {
    return fetch(`${API_URL}/polygon/getimage`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.blob())
        .then(blob => URL.createObjectURL(blob))
}

export function submitBoxSelection(points: {x: number, y:number}[]) {
    return fetch(`${API_URL}/polygon/points`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(points)
    })
        .then(response => response.json())
        .then(data => data)
}

export function submitFeatureHuntImage(file: File): Promise<string[]> {
    const formData = new FormData()
    formData.append("file", file)
    return fetch(`${API_URL}/polygon/image`, {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => data)
}

export function getCluster(layer: string): Promise<{ labels: number[], centers: number[][], distances: number[], outliers: number[]}> {
    return fetch(`${API_URL}/analysis/layer/${layer}/cluster?` + new URLSearchParams({
        outlier_threshold: '2',
    }))
        .then(response => response.json())
        .then(data => data)
}

export function getActivationsImages(node: Node, startFilter: number, nFilters: number, nImgs: number): Promise<string[]> {
    const imgLayerTypes = ["Conv2D", "MaxPooling2D", "AveragePooling2D", "Conv2d", "Cat", "Add"]
    if(!imgLayerTypes.includes(node.layer_type)) {
        return Promise.resolve([])
    }
    
    const promises: Promise<string>[] = []
    Array.from(Array(nImgs).keys()).forEach(imgIdx => {
        Array.from(Array(nFilters).keys(), x => x + startFilter).forEach(filterIdx => {
            promises.push(fetch(`${API_URL}/analysis/image/${imgIdx}/layer/${node.name}/filter/${filterIdx}`)
                .then(response => response.blob())
                .then(blob => URL.createObjectURL(blob))
            )
        })
    })
    
    return Promise.all(promises)
}

export function getAnalysisLayerCoords(node: string): Promise<[number, number][]> {
    return fetch(`${API_URL}/analysis/layer/${node}/embedding?` + new URLSearchParams({
        normalization: "none", // none, row, col
        method: 'mds', // mds, tsne
        distance: 'euclidean' // jaccard, euclidean
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

export function analyze(labels: number[], examplePerClass: number, shuffled: boolean): Promise<{ message: string, task_id: string }> {
    return fetch(`${API_URL}/analysis?examplePerClass=${examplePerClass}&shuffle=${shuffled}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(labels)
    })
    .then(response => response.json())
    .then(data => data)
}

export function getTaskStatus(task_id: string): Promise<{ message: string, task_id: string, payload: null | AnalysisConfig }> {
    return fetch(`${API_URL}/taskStatus?task_id=${task_id}`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
    .then(response => response.json())
    .then(data => data)
}

export function getPredictions(): Promise<number[]> {
    return fetch(`${API_URL}/analysis/predictions`)
        .then(response => response.json())
        .then(data => data)
}

export function getDenseArgmax(layer: string): Promise<number[]> {
    return fetch(`${API_URL}/analysis/layer/${layer}/argmax`)
        .then(response => response.json())
        .then(data => data)
}


export function getInputImages(imgIdxs: number[]): Promise<string[]> {
    if(imgIdxs.length === 0) {
        return Promise.resolve([])
    }
    return Promise.all(imgIdxs.map(
        i => fetch(`${API_URL}/analysis/images/${i}`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        })
            .then(response => response.blob())
            .then(blob => URL.createObjectURL(blob))
        ))
}

export function getActivationOverlay(imgIdxs: number[], node: string, channel: number): Promise<string[]> {
    return Promise.all(imgIdxs.map(
        i => fetch(`${API_URL}/analysis/layer/${node}/${channel}/heatmap/${i}`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        })
            .then(response => response.blob())
            .then(blob => URL.createObjectURL(blob))))
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

export function getConfiguration(): Promise<AnalysisConfig> {
    return fetch(`${API_URL}/loaded_analysis`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => ({
            selectedClasses: data.selectedClasses,
            examplePerClass: data.examplePerClass,
            selectedImages: [],
            shuffled: data.shuffled,
            predictions: data.predictions,
        }))
}
