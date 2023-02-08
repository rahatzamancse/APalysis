import { CurrentModel } from "./features/modelSlice";
import { ModelGraph, Prediction } from "./types";

const API_URL = "http://127.0.0.1:8000/api"

export function getModelGraph(modelName: string = "inception_v3", pretrainWeights: string = "imagenet"): Promise<ModelGraph> {

    return fetch(`${API_URL}/models/?model_name=${modelName}&pretrain_weights=${pretrainWeights}`, {
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

// Function to submit an image to the server
export function submitImage(file: File): Promise<Prediction> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return fetch(`${API_URL}/exampleimg/`, {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => data)
}

export function getActivation(layerName: string): Promise<CurrentModel['value']> {
    return fetch(`${API_URL}/activations/${layerName}`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => ({
            selectedNode: null,
            nFilters: data.n_filters,
            threshold: data.img_summary.reduce((a:number,b:number) => a+b) / data.img_summary.length,
            imgSummary: data.img_summary
        }))
}