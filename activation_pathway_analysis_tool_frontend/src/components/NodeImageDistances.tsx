import React from 'react'
import ScatterPlot from './ScatterPlot'
import { Node } from '../types'
import * as api from '../api'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice';

function NodeImageDistances({ node }: { node: Node }) {
    const analysisResult = useAppSelector(selectAnalysisResult)
    const [coords, setCoords] = React.useState<[number, number][]>([])
    React.useEffect(() => {
        api.getAnalysisLayerCoords(node.name).then((res) => {
            setCoords(res)
        })
    }, [node])

    return analysisResult.coords.length>0?<ScatterPlot coords={coords} labels={analysisResult.labels} width={200} height={200} />:null
}

export default NodeImageDistances