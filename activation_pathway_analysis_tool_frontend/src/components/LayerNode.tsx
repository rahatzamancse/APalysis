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

function LayerNode({ id, data }: { id: string, data: Node }) {

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: NodeColors[data.layer_type],
            borderRadius: 10,
            border: '1px solid #aaa',
        }}>
            <Handle type="target" position={Position.Top} />
            <Card style={{
                width: '25rem',
            }}>
                <Card.Body>
                    <Card.Title>{data.name}</Card.Title>
                    <NodeActivationHeatmap node={data} width={200} height={200} />
                    <Accordion alwaysOpen flush>
                        <LazyAccordionItem header="Details" eventKey="0">
                            <ul>
                                <li> <b>Layer :</b> {data.layer_type} </li>
                                <li> <b>Input :</b> ({data.input_shape.toString()}) </li>
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
            <Handle type="source" position={Position.Bottom} />
        </div>
    );
}

export default LayerNode;