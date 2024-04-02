import React from 'react'
import ScatterPlot from './ScatterPlot'
import { Node } from '../types'
import * as api from '../api'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice';

function NodeImageDistances({ node }: { node: Node }) {
    const analysisResult = useAppSelector(selectAnalysisResult)
    const [truePred, setTruePred] = React.useState<boolean[]>([])
    const [coords, setCoords] = React.useState<[number, number][]>([])
    const [distances, setDistances] = React.useState<number[][]>([])

    React.useEffect(() => {
        api.getAnalysisLayerCoords(node.name, 'umap', 'euclidean', 'none', true).then((res) => {
            setCoords(res)
            api.getPredictions().then((res) => {
                const truePredTmp: boolean[] = []
                analysisResult.selectedClasses.forEach((label, i) => {
                    for(let j=0; j<analysisResult.examplePerClass; j++) {
                        truePredTmp.push(res[i*analysisResult.examplePerClass+j] === label)
                    }
                })
                setTruePred(truePredTmp)
                
                api.getAnalysisDistanceMatrix(node.name)
                    .then(setDistances)
            })
        })
    }, [node, analysisResult.examplePerClass, analysisResult.selectedClasses])
    
    return coords.length>0 && distances.length>0 && truePred.length>0 ? <ScatterPlot
        node={node}
        coords={coords}
        preds={truePred}
        distances={distances}
        labels={analysisResult.selectedClasses.map(label => Array(analysisResult.examplePerClass).fill(label)).flat()}
        width={200}
        height={200}
    /> : <></>
}

export default NodeImageDistances