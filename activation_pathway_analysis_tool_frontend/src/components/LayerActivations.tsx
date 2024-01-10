import React from 'react'
import Card from 'react-bootstrap/Card'
import { Node } from '../types'

import * as api from '../api'
import { Spinner } from 'react-bootstrap'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice';

function LayerActivations({ node }: { node: Node }) {
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const nImgs = analyzeResult.selectedClasses.length * analyzeResult.examplePerClass
    const [currentPage, setCurrentPage] = React.useState<number>(0)
    const [activations, setActivations] = React.useState<string[]>([])
    const examplePerPage = 5
    const nPages = Math.ceil(node.output_shape[3] / examplePerPage)

    React.useEffect(() => {
        if (nImgs === 0) return
        const imgNodes = ['Conv2D', 'Concatenate']
        if (!imgNodes.some(imgNode => node.layer_type.toLowerCase().includes(imgNode.toLowerCase()))) return
        api.getActivationsImages(node,
            currentPage === nPages-1? node.output_shape[3] - (nPages-1)*examplePerPage : currentPage*examplePerPage,
            examplePerPage, nImgs).then(setActivations)
    }, [node, currentPage])
    
    const nextPage = () => {
        if (currentPage === nPages - 1) return
        setCurrentPage(currentPage + 1)
    }
    const prevPage = () => {
        if (currentPage === 0) return
        setCurrentPage(currentPage - 1)
    }

    return activations.length > 0 ? <div style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
        // maxHeight: "200px",
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
        <div style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            marginTop: "10px"
        }}>
            <button onClick={prevPage} disabled={currentPage === 0}>Previous</button>
            {/* Add page numbers */}
            <span style={{padding: "0 10px"}}>{currentPage+1}/{nPages}</span>
            <button onClick={nextPage} disabled={currentPage === nPages-1}>Next</button>
        </div>

    </div > : <Spinner animation="border" role="status" style={{ marginLeft: '40%' }}>
        <span className="visually-hidden">Loading...</span>
    </Spinner>
}

export default LayerActivations