import React from 'react'
import Card from 'react-bootstrap/Card'
import { Node } from '../types'

import * as api from '../api'
import { Spinner } from 'react-bootstrap'

function LayerDetails({ node }: { node: Node }) {

    const [layerDetails, setLayerDetails] = React.useState<{
        nFilters: number,
        threshold: number,
        imgSummary: number[],
        imgUrls: string[],
    }>({
        nFilters: 0,
        threshold: 0,
        imgSummary: [],
        imgUrls: [],
    })

    const getLayerDetails = React.useMemo(() => {
        return api.getActivation(node.name)
            .then((res1) => {
                return api.getActivationsImages(res1.nFilters, node.name)
                    .then((imgUrls) => ({
                        nFilters: res1.nFilters,
                        threshold: res1.threshold,
                        imgSummary: res1.imgSummary,
                        imgUrls: imgUrls,
                    })
                    )
            })
    }, [node])

    React.useEffect(() => {
        const imgNodes = ['Conv2D', 'Concatenate']
        console.log(node.layer_type)
        if (!imgNodes.some(imgNode => node.layer_type.toLowerCase().includes(imgNode.toLowerCase()))) return

        getLayerDetails.then((res) => {
            setLayerDetails(res)
        })
    }, [node])


    return layerDetails.imgUrls.length > 0 ? <div style={{
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
            resize:"both"
        }}>
            {layerDetails.imgUrls.map((image, i) =>
            <Card className="m-1" key={i} style={{
                width: "30%",
                padding: "0px"
            }}>
                <Card.Img variant="top" src={image} style={{
                    opacity: layerDetails.imgSummary[i] >= layerDetails.threshold ? 1 : 0.2,
                    margin: "0",
                    padding: "0",
                }} />
            </Card>)}
        </div>

    </div > : <Spinner animation="border" role="status" style={{ marginLeft: '40%' }}>
        <span className="visually-hidden">Loading...</span>
    </Spinner>
}

export default LayerDetails