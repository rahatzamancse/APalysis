import { AnalysisResult } from "./features/analyzeSlice";
import { ModelGraph, Prediction } from "./types";
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
            nodes: data.nodes.map((node: any) => ({
                id: node.id,
                label: node.label,
                layer_type: node.layer_type,
                name: node.name,
                input_shape: node.input_shape,
                output_shape: node.output_shape,
                tensor_type: node.tensor_type,
                pos: node.pos? { x: node.pos.x, y: node.pos.y } : undefined
            })),
            edges: data.links.map((edge: any) => ({
                source: edge.source,
                target: edge.target
            }))
        }))
}

export function getLabels(): Promise<string[]> {
    return fetch(`${API_URL}/labels/`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => data)
}

export function getActivationsImages(node: Node, nImgs: number): Promise<string[]> {
    const imgLayerTypes = ["Conv2D", "MaxPooling2D", "AveragePooling2D"]
    if(!imgLayerTypes.includes(node.layer_type)) {
        return Promise.resolve([])
    }
    
    const promises: Promise<string>[] = []
    Array.from(Array(node.output_shape[3]).keys()).forEach(filterIdx => {
        Array.from(Array(nImgs).keys()).forEach(imgIdx => {
            promises.push(fetch(`${API_URL}/analysis/image/${imgIdx}/layer/${node.name}/filter/${filterIdx}`)
                .then(response => response.blob())
                .then(blob => URL.createObjectURL(blob))
            )
        })
    })
    
    return Promise.all(promises)
}

export function getAnalysisLayerCoords(node: string): Promise<[number, number][]> {
    return fetch(`${API_URL}/analysis/layer/${node}/embedding`)
        .then(response => response.json())
        .then(data => data)
}

export function getAnalysisHeatmap(node: string): Promise<number[][]> {
    return fetch(`${API_URL}/analysis/layer/${node}/heatmap`)
        .then(response => response.json())
        .then(data => data)
}

export function analyze(labels: number[], examplePerClass: number): Promise<AnalysisResult> {
    return fetch(`${API_URL}/analysis?examplePerClass=${examplePerClass}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(labels)
    })
        .then(response => response.json())
        .then(data => data)
}

export function getInputImages(imgIdxs: number[]): Promise<string[]> {
    return Promise.all(imgIdxs.map(
        i => fetch(`${API_URL}/analysis/images/${i}`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        })
            .then(response => response.blob())
            .then(blob => URL.createObjectURL(blob))))
}
