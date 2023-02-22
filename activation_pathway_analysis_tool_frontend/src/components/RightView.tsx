import React from 'react'
import Scatter from './Scatter'
import { useAppSelector } from '../app/hooks'
import { selectAnalysisResult } from '../features/analyzeSlice';

function RightView() {

    const analysisResult = useAppSelector(selectAnalysisResult)

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
        {analysisResult.coords.length>0?<Scatter width={260} height={260} coords={analysisResult.coords} labels={analysisResult.labels} />:null}
    </div>

}

export default RightView