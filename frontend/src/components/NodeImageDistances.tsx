import React from 'react'
import ScatterPlot from './ScatterPlot'
import { Node } from '../types'
import * as api from '../api'

function NodeImageDistances({ node }: { node: Node }) {
    const [coords, setCoords] = React.useState<[number, number][]>([])
    const [distances, setDistances] = React.useState<number[][]>([])

    React.useEffect(() => {
        api.getAnalysisLayerCoords(node.name, 'umap', 'euclidean', 'none', true).then(setCoords)
        api.getAnalysisDistanceMatrix(node.name).then(setDistances)
    }, [node])
    
    return coords.length>0 && distances.length>0 ? <ScatterPlot
        node={node}
        coords={coords}
        distances={distances}
        width={200}
        height={200}
    /> : <></>
}

export default NodeImageDistances