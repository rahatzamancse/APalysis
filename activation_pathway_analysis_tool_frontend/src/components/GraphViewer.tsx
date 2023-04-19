import React from 'react'
import * as api from '../api'

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
    ReactFlowInstance,
    useReactFlow,
    useUpdateNodeInternals,
    ControlButton,
    XYPosition,
} from 'reactflow';
import { toPng } from 'html-to-image';

import 'reactflow/dist/style.css';
import LayerNode from './LayerNode';
import { Node as BaseNode } from '../types'

const initialNodes: Node[] = [
  { id: '1', position: { x: 0, y: 0 }, data: { label: 'Loading Graph\nPlease Wait' } },
];
const initialEdges: Edge[] = [];

const GRAPH_HEIGHT_FACTOR = 200
const GRAPH_WIDTH_FACTOR = 400

const nodeTypes = { layerNode: LayerNode };

function GraphViewer() {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const [layoutHorizontal, setLayoutHorizontal] = React.useState(true);
    
    const reactFlowInstance = useReactFlow();
    const updateNodeInternals = useUpdateNodeInternals()
    const flowRef = React.useRef<HTMLDivElement>(null);
    
    React.useEffect(() => {
        api.getModelGraph().then(modelGraph => {
            const max_depth = modelGraph.meta.depth * 2.5
            
            // TODO: Get the width of the model tree
            const max_width = 20
            
            const getX = (node: BaseNode) => node.pos?node.pos.y*-1*GRAPH_HEIGHT_FACTOR*max_depth:0
            const getY = (node: BaseNode) => node.pos?node.pos.x*GRAPH_WIDTH_FACTOR*max_width:0
            
            setNodes(modelGraph.nodes.map(node => ({
                id: node.id,
                position: { 
                    x: layoutHorizontal?getX(node):getY(node),
                    y: layoutHorizontal?getY(node):getX(node), 
                },
                data: {
                    label: node.label,
                    layer_type: node.layer_type,
                    name: node.name,
                    input_shape: node.input_shape,
                    kernel_size: node.kernel_size,
                    output_shape: node.output_shape,
                    tensor_type: node.tensor_type,
                    layout_horizontal: layoutHorizontal
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
    
    React.useEffect(() => {
        setNodes(val => val.map(node => ({
                ...node,
                position: {
                    ...node.position,
                    x: node.position.y,
                    y: node.position.x
                },
                data: {
                    ...node.data,
                    layout_horizontal: !layoutHorizontal
                }
            }))
        )
        
        nodes.forEach(node => {
            updateNodeInternals(node.id)
        })
        
        setTimeout(() => {
            reactFlowInstance.fitView({ duration: 800 })
        }, 1000)
        
    }, [layoutHorizontal])
    

    const positions = nodes.map(node => node.position)
    const minX = Math.min(...positions.map(pos => pos.x)) - 500
    const minY = Math.min(...positions.map(pos => pos.y)) - 500
    const maxX = Math.max(...positions.map(pos => pos.x)) + 500
    const maxY = Math.max(...positions.map(pos => pos.y)) + 500
    const maxRange = Math.max((maxX-minX)/2, (maxY-minY)/2)
    const center = { x: minX + (maxX-minX)/2, y: minY + (maxY-minY)/2 }
    const translationExtent: [[number,number],[number,number]] = [
        [center.x - maxRange, center.y - maxRange],
        [center.x + maxRange, center.y + maxRange]
    ]
    
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
            minZoom={0.1}
            maxZoom={10}
            translateExtent={translationExtent}
            elevateEdgesOnSelect
            ref={flowRef}
        >
            <MiniMap pannable zoomable style={{
                border: '1px solid #000',
            }}/>
            <Controls >
                <ControlButton onClick={() => setLayoutHorizontal(val => !val)}>
                    {layoutHorizontal ? "H" : "V"}
                </ControlButton>
                <ControlButton onClick={() => {
                    if(flowRef.current === null) return
                    toPng(flowRef.current, {
                        filter: node => !(
                            node?.classList?.contains('react-flow__minimap') ||
                            node?.classList?.contains('react-flow__controls')
                        ),
                        // canvasWidth: 5000,
                        // canvasHeight: 1000,
                    }).then(dataUrl => {
                        const a = document.createElement('a');
                        a.setAttribute('download', 'reactflow.png');
                        a.setAttribute('href', dataUrl);
                        a.click();
                    });

                }}>
                    <img src="assets/export.png" alt="Export" width="16px" height="16px" />
                </ControlButton>
            </Controls>
            <Background />
        </ReactFlow>
    </div>
}

export default GraphViewer