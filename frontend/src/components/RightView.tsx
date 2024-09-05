import React from 'react'
import ScatterPlot from './ScatterPlot'
import * as api from '../api'

function RightView() {
    // const analysisResult = useAppSelector(selectAnalysisResult)
    // const [coords, setCoords] = React.useState<[number, number][]>([])
    // const [truePred, setTruePred] = React.useState<boolean[]>([])
    // const [distances, setDistances] = React.useState<number[][]>([])
    
    // React.useEffect(() => {
    //     if(analysisResult.selectedImages.length === 0) return
    // }, [analysisResult.selectedImages])
    
    // React.useEffect(() => {
    //     if(analysisResult.examplePerClass === 0) return
    //     api.getAllEmbedding().then(coords => {
    //         setCoords(coords)
    //         api.getAllDistances().then(setDistances)
    //     })
    // }, [analysisResult])
            
    
    // React.useEffect(() => {
    //         api.getPredictions().then((res) => {
    //             const truePredTmp: boolean[] = []
    //             analysisResult.selectedClasses.forEach((label, i) => {
    //                 for(let j=0; j<analysisResult.examplePerClass; j++) {
    //                     truePredTmp.push(res[i*analysisResult.examplePerClass+j] === label)
    //                 }
    //             })
    //             setTruePred(truePredTmp)
    //         })
    // }, [analysisResult])

    return <div className="rsection" style={{
        display: "flex",
        flexDirection: "column",
        width: "20%",
        minWidth: "300px",
        maxWidth: "500px",
        height: "90vh",
        padding: "20px",
    }}>
        {/* <h5>Activation Pathway Summary</h5>
        {coords.length>0?<ScatterPlot
            node={null}
            width={260}
            height={260}
            coords={coords}
            distances={distances}
            labels={analysisResult.selectedClasses.map(label => Array(analysisResult.examplePerClass).fill(label)).flat()}
            preds={truePred}
        />:null} */}
    </div>

}

export default RightView