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
import LayerOutEdges from './LayerOutEdges';


function LayerNode({ id, data }: { id: string, data: Node }) {
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
            <Card className={data.tutorial_node?'tutorial-cnn-layer':''}>
                <Card.Body style={{
                    display: 'flex',
                    flexDirection: 'column',
                }}>
                    <Card.Title>{data.name}</Card.Title>
                    <Accordion alwaysOpen flush>
                        <LazyAccordionItem className={data.tutorial_node?'tutorial-cnn-layer-details':''} header="Details" eventKey="2">
                            <ul>
                                <li> <b>Layer :</b> {data.layer_type} </li>
                                <li> <b>Input :</b> ({data.input_shape.toString()}) </li>
                                {data.kernel_size?<li> <b>Kernel Shape :</b> ({data.kernel_size.toString()}) </li>:null}
                                {data.out_edge_weight && <li> <b>Kernel # :</b> {data.out_edge_weight.length} </li>}
                                <li> <b>Output :</b> ({data.output_shape.toString()}) </li>
                            </ul>
                        </LazyAccordionItem>
                        {['Conv2D', 'Concatenate', 'Conv2d', 'Cat'].includes(data.layer_type) && <LazyAccordionItem header="Activations" eventKey="3">
                            <LayerActivations node={data} />
                        </LazyAccordionItem>}
                        {['Dense', 'Linear'].includes(data.layer_type) && <LazyAccordionItem header="Predictions" eventKey="5">
                            <DenseArgmax node={data} />
                        </LazyAccordionItem>}
                        {/* {['Conv2D'].includes(data.layer_type) && <LazyAccordionItem header="Edges" eventKey="6">
                            <LayerOutEdges node={data} />
                        </LazyAccordionItem>} */}
                        {['Conv2D', 'Dense', 'Concatenate', 'Conv2d', 'Linear', 'Cat', 'Add'].includes(data.layer_type) && <LazyAccordionItem header="Scatterplot View" headerColor='#2853c9' eventKey="4">
                            <NodeImageDistances node={data} />
                        </LazyAccordionItem>}
                        {['Conv2D', 'Dense', 'Concatenate', 'Conv2d', 'Linear', 'Cat', 'Add'].includes(data.layer_type) && <LazyAccordionItem header="Jaccard Similarity View" headerColor='#b2006d' eventKey="1">
                            <NodeActivationMatrix node={data} width={350} height={350} />
                        </LazyAccordionItem>}
                        {[
                            // tensorflow
                            'Conv2D', 'Dense', 'Concatenate',
                            // pytorch
                            'Conv2d', 'Linear', 'Cat', 'Add',
                        ].includes(data.layer_type) && <LazyAccordionItem className={data.tutorial_node?'tutorial-cnn-layer-heatmap':''} header="Heatmap View" eventKey="0" headerColor='#00724a'>
                            <NodeActivationHeatmap
                                node={data}
                                minWidth={350}
                                // height={data.output_shape[data.output_shape.length-1]!*HEATMAP_HEIGHT_FACTOR}
                                minHeight={300}
                                normalizeRow={true}
                            />
                        </LazyAccordionItem>}
                    </Accordion>
                </Card.Body>
            </Card>
            <Handle type="source" position={data.layout_horizontal?Position.Right:Position.Bottom} />
        </div>
    );
}

export default LayerNode;