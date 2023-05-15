import React from 'react';
import { Handle, Position } from 'reactflow';
import { Node } from '../types'
import { NodeColors } from '../utils';
import { Card, Accordion } from 'react-bootstrap';
import LayerActivations from './LayerActivations';
import LazyAccordionItem from './LazyAccordionItem';
import NodeImageDistances from './NodeImageDistances';
import NodeActivationHeatmap from './NodeActivationHeatmap';
import NodeActivationMatrix from './NodeActivationMatrix';
import DenseArgmax from './DenseArgmax';
import { useAppSelector } from '../app/hooks';
import analyzeSlice, { selectAnalysisResult, setAnalysisResult } from '../features/analyzeSlice';



function LayerNode({ id, data }: { id: string, data: Node }) {
    const analysisResult = useAppSelector(selectAnalysisResult)
    
    const HEATMAP_HEIGHT_FACTOR = 5
    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: NodeColors[data.layer_type],
            borderRadius: 10,
            border: '1px solid #aaa',
        }}>
            <Handle type="target" position={data.layout_horizontal?Position.Left:Position.Top} />
            <Card>
                <Card.Body style={{
                    display: 'flex',
                    flexDirection: 'column',
                }}>
                    <Card.Title>{data.name}</Card.Title>
                    <Accordion alwaysOpen flush>
                        <LazyAccordionItem header="Details" eventKey="2">
                            <ul>
                                <li> <b>Layer :</b> {data.layer_type} </li>
                                <li> <b>Input :</b> ({data.input_shape.toString()}) </li>
                                <li> <b>Kernel Shape :</b> ({data.kernel_size.toString()}) </li>
                                <li> <b>Output :</b> ({data.output_shape.toString()}) </li>
                            </ul>
                        </LazyAccordionItem>
                        {analysisResult.examplePerClass !== 0 && ['Conv2D', 'Dense', 'Concatenate'].includes(data.layer_type) && <LazyAccordionItem header="Activation Heatmap" eventKey="0">
                            <NodeActivationHeatmap
                                node={data}
                                width={350}
                                height={data.output_shape[data.output_shape.length-1]!*HEATMAP_HEIGHT_FACTOR}
                                normalizeRow={true}
                                sortby={data.layer_type === 'Dense' ? 'none' : 'count'}
                            />
                        </LazyAccordionItem>}
                        {analysisResult.examplePerClass !== 0 && ['Dense'].includes(data.layer_type) && <LazyAccordionItem header="Argmax" eventKey="5">
                            <DenseArgmax node={data} />
                        </LazyAccordionItem>}
                        {analysisResult.examplePerClass !== 0 && ['Conv2D', 'Dense', 'Concatenate'].includes(data.layer_type) && <LazyAccordionItem header="Activation Jaccard Similarity" eventKey="1">
                            <NodeActivationMatrix node={data} width={350} height={350} />
                        </LazyAccordionItem>}
                        {analysisResult.examplePerClass !== 0 && ['Conv2D', 'Concatenate'].includes(data.layer_type) && <LazyAccordionItem header="Activations" eventKey="3">
                            <LayerActivations node={data} />
                        </LazyAccordionItem>}
                        {analysisResult.examplePerClass !== 0 && ['Conv2D', 'Dense', 'Concatenate'].includes(data.layer_type) && <LazyAccordionItem header="Activation Distances" eventKey="4">
                            <NodeImageDistances node={data} />
                        </LazyAccordionItem>}
                    </Accordion>
                </Card.Body>
            </Card>
            <Handle type="source" position={data.layout_horizontal?Position.Right:Position.Bottom} />
        </div>
    );
}

export default LayerNode;