import React from 'react'
import ScatterPlot from './ScatterPlot'
import { useAppSelector } from '../app/hooks'
import * as api from '../api'
import { selectAnalysisResult } from '../features/analyzeSlice';

function RightView() {
    const analysisResult = useAppSelector(selectAnalysisResult)
    const [selectedImgs, setSelectedImgs] = React.useState<string[]>([])
    const [coords, setCoords] = React.useState<[number, number][]>([])
    const [truePred, setTruePred] = React.useState<boolean[]>([])
    
    React.useEffect(() => {
        if(analysisResult.selectedImages.length === 0) return
        api.getInputImages(analysisResult.selectedImages).then(setSelectedImgs)
    }, [analysisResult.selectedImages])
    
    React.useEffect(() => {
        if(analysisResult.examplePerClass === 0) return
        api.getAllEmbedding().then(setCoords)
    }, [analysisResult])
            
    
    React.useEffect(() => {
            api.getPredictions().then((res) => {
                const truePredTmp: boolean[] = []
                analysisResult.selectedClasses.forEach((label, i) => {
                    for(let j=0; j<analysisResult.examplePerClass; j++) {
                        truePredTmp.push(res[i*analysisResult.examplePerClass+j] == label)
                    }
                })
                setTruePred(truePredTmp)
            })
    }, [analysisResult])

    return <div className="rsection" style={{
        display: "flex",
        flexDirection: "column",
        width: "20%",
        minWidth: "300px",
        maxWidth: "500px",
        height: "90vh",
        padding: "20px",
    }}>
        <h5>Activation Pathway Summary</h5>
        {coords.length>0?<ScatterPlot
            width={260}
            height={260}
            coords={coords}
            labels={analysisResult.selectedClasses.map(label => Array(analysisResult.examplePerClass).fill(label)).flat()}
            preds={truePred}
        />:null}
        {selectedImgs.length>0?<div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
            <h5>Selected Images</h5>
            <div style={{display: "flex", flexDirection: "row", flexWrap: "wrap", justifyContent: "center"}}>
                {selectedImgs.map((img, i) => {
                    return <img key={i} src={img} width={100} height={100} style={{margin: "5px"}} />
                })}
            </div>
        </div>:null}

    </div>

}

export default RightView