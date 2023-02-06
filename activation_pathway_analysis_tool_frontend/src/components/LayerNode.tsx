import React from 'react';
import { Handle, NodeProps, Position } from 'reactflow';
import { Node } from '../types'
import { NodeColors } from '../utils';
import * as api from '../api'
import { useAppDispatch } from '../app/hooks'
import { setSelectedNode } from '../features/modelSlice'

function LayerNode({ id, data }: { id: string, data: Node }) {

    const dispatch = useAppDispatch();

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: NodeColors[data.layer_type],
            borderRadius: 10,
            border: '1px solid #aaa',
        }}>
            <Handle type="target" position={Position.Top} />
            <h5 style={{ alignSelf: 'center' }}>{data.name}</h5>
            <ul>
                <li> <b>Layer :</b> {data.layer_type} </li>
                <li> <b>Input :</b> ({data.input_shape.toString()}) </li>
                <li> <b>Output :</b> ({data.output_shape.toString()}) </li>
            </ul>
            <div style={{ padding: 20 }}>
                <button
                    onClick={() => {
                        api.getActivation(data.name)
                            .then((res) => {
                                dispatch(setSelectedNode({
                                    selectedNode: data.name,
                                    nFilters: res.n_filters,
                                }))
                            })
                    }}
                    className='nodrag'
                >Show Details</button>
            </div>
            <Handle type="source" position={Position.Bottom} />
        </div>
    );
}

export default LayerNode;