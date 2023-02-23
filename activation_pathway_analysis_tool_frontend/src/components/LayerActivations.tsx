import React from 'react'
import Card from 'react-bootstrap/Card'
import { Node } from '../types'

import * as api from '../api'
import { Spinner } from 'react-bootstrap'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice';

function LayerActivations({ node }: { node: Node }) {
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const nImgs = analyzeResult.labels.length
    const [activations, setActivations] = React.useState<string[]>([])

    React.useEffect(() => {
        if (nImgs === 0) return
        const imgNodes = ['Conv2D', 'Concatenate']
        if (!imgNodes.some(imgNode => node.layer_type.toLowerCase().includes(imgNode.toLowerCase()))) return
        api.getActivationsImages(node, nImgs).then(setActivations)
    }, [node])

    return activations.length > 0 ? <div style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
        maxHeight: "200px",
        overflowY: "scroll",
        overflowX: "hidden",
    }}>
        <div style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
        }}>
            {activations.map((image, i) => <img key={i} src={image} style={{
                flexBasis: `${100 / nImgs}%`,
                maxWidth: `${(100 / nImgs)}%`,
                margin: "0"
            }} />)}
        </div>

    </div > : <Spinner animation="border" role="status" style={{ marginLeft: '40%' }}>
        <span className="visually-hidden">Loading...</span>
    </Spinner>
}

export default LayerActivations