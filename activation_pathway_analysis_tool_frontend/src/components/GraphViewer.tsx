import React from 'react'
import * as api from '../api'
import { ModelGraph } from '../types'

import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  FitViewOptions,
  applyNodeChanges,
  applyEdgeChanges,
  Node,
  Edge,
  NodeChange,
  EdgeChange,
  Connection,
} from 'reactflow';


import 'reactflow/dist/style.css';
import LayerNode from './LayerNode';

const initialNodes: Node[] = [
  { id: '1', position: { x: 0, y: 0 }, data: { label: 'Loading Graph\nPlease Wait' } },
];
const initialEdges: Edge[] = [];

const GRAPH_HEIGHT_FACTOR = 200
const GRAPH_WIDTH_FACTOR = 400

const nodeTypes = { layerNode: LayerNode };

function GraphViewer() {
    // const [graph, setGraph] = React.useState<ModelGraph | null>(null)

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    React.useEffect(() => {
        api.getModelGraph().then(modelGraph => {
            // setGraph(modelGraph)
            const input_layer_name = modelGraph.nodes.find(node => node.layer_type === 'InputLayer')?.name

            // TODO: Assign depth to each node in modelGraph
            
            // TODO: Get the maximum depth of modelGraph
            const max_depth = 65
            const max_width = 5

            setNodes(modelGraph.nodes.map(node => ({
                id: node.id,
                position: { x: node.pos?node.pos.x*GRAPH_WIDTH_FACTOR*max_width:0, y: node.pos?node.pos.y*-1*GRAPH_HEIGHT_FACTOR*max_depth:0 },
                data: {
                    label: node.label,
                    layer_type: node.layer_type,
                    name: node.name,
                    input_shape: node.input_shape,
                    output_shape: node.output_shape,
                    tensor_type: node.tensor_type,
                },
                type: 'layerNode',
            })))
            setEdges(modelGraph.edges.map(edge => ({
                id: `${edge.source}-${edge.target}`,
                source: edge.source,
                target: edge.target,
                animated: true,

                label: ''
            })))
        })
    }, [])


    return <div className="rsection" style={{
        display: "flex",
        width: "100%",
        minWidth: "600px",
        height: "90vh",
        minHeight: "1000px",
    }}>
        <ReactFlow
            nodes={nodes}
            nodeTypes={nodeTypes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            fitViewOptions={{ padding: 0.1 }}
            attributionPosition="bottom-right"
        >
            <MiniMap pannable zoomable style={{
                border: '1px solid #000',
            }}/>
            <Controls />
            <Background />
        </ReactFlow>
    </div>
}

export default GraphViewer