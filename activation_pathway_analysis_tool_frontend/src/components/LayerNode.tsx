import React from 'react';
import { Handle, NodeProps, Position } from 'reactflow';
import { Node } from '../types'
import { NodeColors } from '../utils';
import * as api from '../api'
import { useAppDispatch } from '../app/hooks'
import { Card, Button, Accordion } from 'react-bootstrap';
import LayerActivations from './LayerActivations';
import { useAppSelector } from '../app/hooks'
import { current } from '@reduxjs/toolkit';
import LazyAccordionItem from './LazyAccordionItem';
import ScatterPlot from './ScatterPlot';
import NodeImageDistances from './NodeImageDistances';
import NodeActivationHeatmap from './NodeActivationHeatmap';
import NodeActivationMatrix from './NodeActivationMatrix';



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
            <Card>
                <Card.Body style={{
                    display: 'flex',
                    flexDirection: 'column',
                }}>
                    <Card.Title>{data.name}</Card.Title>
                    <NodeActivationHeatmap
                        node={data}
                        width={350}
                        height={data.output_shape[data.output_shape.length-1]!*HEATMAP_HEIGHT_FACTOR}
                        normalizeRow={false}
                        sortby={data.layer_type === 'Dense' ? 'none' : 'count'}
                    />
                    <NodeActivationMatrix node={data} width={350} height={350} />
                    <Accordion alwaysOpen flush>
                        <LazyAccordionItem header="Details" eventKey="0">
                            <ul>
                                <li> <b>Layer :</b> {data.layer_type} </li>
                                <li> <b>Input :</b> ({data.input_shape.toString()}) </li>
                                <li> <b>Kernel Shape :</b> ({data.kernel_size.toString()}) </li>
                                <li> <b>Output :</b> ({data.output_shape.toString()}) </li>
                            </ul>
                        </LazyAccordionItem>
                        <LazyAccordionItem header="Activations" eventKey="1">
                            <LayerActivations node={data} />
                        </LazyAccordionItem>
                        <LazyAccordionItem header="Activation Distances" eventKey="2">
                            <NodeImageDistances node={data} />
                        </LazyAccordionItem>
                    </Accordion>
                </Card.Body>
            </Card>
            <Handle type="source" position={data.layout_horizontal?Position.Right:Position.Bottom} />
        </div>
    );
}

export default LayerNode;