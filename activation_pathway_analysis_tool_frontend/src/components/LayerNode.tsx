import React from 'react';
import { Handle, NodeProps, Position } from 'reactflow';
import { Node } from '../types'
import { NodeColors } from '../utils';
import * as api from '../api'
import { useAppDispatch } from '../app/hooks'
import { setSelectedNode } from '../features/modelSlice'
import { Card, Button, Accordion } from 'react-bootstrap';
import LayerDetails from './LayerDetails';
import { useAppSelector } from '../app/hooks'
import { selectCurrentModel } from '../features/modelSlice'
import { current } from '@reduxjs/toolkit';
import LazyAccordionItem from './LazyAccordionItem';

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
                    <Accordion alwaysOpen flush>
                        <LazyAccordionItem header="Details" eventKey="0">
                            <ul>
                                <li> <b>Layer :</b> {data.layer_type} </li>
                                <li> <b>Input :</b> ({data.input_shape.toString()}) </li>
                                <li> <b>Output :</b> ({data.output_shape.toString()}) </li>
                            </ul>
                        </LazyAccordionItem>
                        <LazyAccordionItem header="Activations" eventKey="1">
                            <LayerDetails node={data} />
                        </LazyAccordionItem>
                    </Accordion>
                </Card.Body>
            </Card>
            <Handle type="source" position={Position.Bottom} />
        </div>
    );
}

export default LayerNode;