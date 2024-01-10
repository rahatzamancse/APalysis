import React from 'react'
import Form from 'react-bootstrap/Form';
import * as api from '../api'
import { useAppDispatch, useAppSelector } from '../app/hooks'
import { selectAnalysisResult, setAnalysisResult } from '../features/analyzeSlice';
import { chunkify } from '../utils';
import { Modal, Spinner } from 'react-bootstrap';
import ProgressModal from './ProgressModal';



function Controls() {
    const [isProcessing, setIsProcessing] = React.useState<boolean>(false)
    const [processingMessage, setProcessingMessage] = React.useState<string>("")
    const [uploadOwn, setUploadOwn] = React.useState<boolean>(false)
    const [classes, setClasses] = React.useState<string[]>([])
    const analysisResult = useAppSelector(selectAnalysisResult)
    const [inputImages, setInputImages] = React.useState<string[]>([])
    const [shuffled, setShuffled] = React.useState<boolean>(false)
    const [nExamplePerClass, setNExamplePerClass] = React.useState<number>(5)

    const checkTaskStatus = async (task_id: string) => {
        setIsProcessing(true)
        setTimeout(() => {
            api.getTaskStatus(task_id).then((res) => {
                if(res.payload === null) {
                    setProcessingMessage(res.message)
                    checkTaskStatus(task_id)
                } else {
                    setIsProcessing(false)
                    setProcessingMessage(res.message)

                    dispatch(setAnalysisResult(res.payload))
                    api.getInputImages([...Array(res.payload.selectedClasses.length*res.payload.examplePerClass).keys()]).then(setInputImages)
                    setNExamplePerClass(res.payload.examplePerClass)
                    if(nExamplePerClassRef.current !== null) nExamplePerClassRef.current.value = res.payload.examplePerClass.toString()
                    setShuffled(res.payload.shuffled)
                }
            })
        }, 3000)
    }
    
    const checkboxRefs = React.useRef<HTMLInputElement[]>([])
    const nExamplePerClassRef = React.useRef<HTMLInputElement>(null)

    const dispatch = useAppDispatch()

    React.useEffect(() => {
        api.analyze([0, 1, 2, 3], 2, true)
            .then(({ message, task_id }) => {
                checkTaskStatus(task_id)
            })
        api.getLabels().then(setClasses)
    }, [])
    
    React.useEffect(() => {
        api.getConfiguration().then((res) => {
            checkboxRefs.current.forEach((checkbox, index) => {
                checkbox.checked = res.selectedClasses.includes(index)
            })
        })
    }, [classes])
    
    return <><div className="rsection" style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        width: "20%",
        minWidth: "300px",
        maxWidth: "500px",
        minHeight: "90vh",
        maxHeight: "90vh",
        padding: "20px",
    }}>
        <Form>
            {/* <div className="mb-3">
                <Form.Check
                    type="switch"
                    id="custom-switch"
                    label="Upload your own image"
                    checked={uploadOwn}
                    onChange={e => setUploadOwn(e.target.checked)}
                />
            </div> */}
            <div style={{ alignSelf: "flex-end", margin: "10px" }}>
                <button title='Reload the last loaded state of the Neural Network' type="button" className="btn btn-primary" onClick={(e) => {
                    if(checkboxRefs.current.length === 0) return
                    api.getConfiguration().then((res) => {
                        setNExamplePerClass(res.examplePerClass)
                        if(nExamplePerClassRef.current !== null) nExamplePerClassRef.current.value = res.examplePerClass.toString()
                        checkboxRefs.current.forEach((checkbox, index) => {
                            checkbox.checked = res.selectedClasses.includes(index)
                        })
                        setShuffled(res.shuffled)
                        api.getInputImages([...Array(res.selectedClasses.length*res.examplePerClass).keys()]).then(setInputImages)
                        dispatch(setAnalysisResult(res))
                    })
                }}><span className="glyphicon glyphicon-refresh">Refresh</span></button>
            </div>
        </Form>
        {uploadOwn ? <h5>Stub Upload own image</h5> : <>
            <h5 className="mb-3">Select Labels to Analyze</h5>
            <Form.Label htmlFor="nExamplePerClass" style={{ float: 'left', width: '40%'}}>Image per Class</Form.Label> <Form.Check type="switch" id="shufflecheck" style={{float: 'right', width: '30%'}} label="Shuffle" checked={shuffled} onChange={e => setShuffled(e.target.checked)} />
            <Form.Control ref={nExamplePerClassRef} id="nExamplePerClass" className="mb-5" type="number" min={1} max={50} />
            <div style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                justifyContent: "flex-start",
                overflowY: "scroll",
                height: "30vh",
                width: "100%",
            }}>
                {classes.map((label, index) => (
                    <Form.Check key={index} type="checkbox" label={label} id={`checkbox_${index}`} ref={(ref: HTMLInputElement) => {checkboxRefs.current[index] = ref}} />
                ))}
            </div>
            <button className="btn btn-primary mt-5" onClick={() => {
                const checkedLabels = checkboxRefs.current.map((checkbox, index) => checkbox.checked).reduce<number[]>((out, bool, index) => bool?out.concat(index):out, [])
                // const checkedLabels = Array.from(document.querySelectorAll("input[type=checkbox]:checked")).map(checkbox => parseInt(checkbox.id.split("_")[1]))
                
                const currentNExamplePerClass = parseInt(nExamplePerClassRef.current?.value || "5")

                api.analyze(checkedLabels, currentNExamplePerClass, shuffled)
                    .then(({ message, task_id }) => {
                        checkTaskStatus(task_id)
                    })
            }}>Analyze</button>
        </>}
        
        {inputImages.length > 0 ? <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            justifyContent: "flex-start",
            width: "100%",
            overflowY: "scroll",
        }}>
            <h4 className="mt-5">Input Images</h4>
            {chunkify(inputImages, nExamplePerClass).map((chunk, i) => <div key={'outerdiv'+i} style={{
                width: "100%",
                marginBottom: "20px",
            }}>
                <h5 key={'h5'+i}>Class: {classes[analysisResult.selectedClasses[i]]}</h5>
                <div 
                    key={'div'+i}
                    style={{
                        display: "flex",
                        flexDirection: "row",
                        alignItems: "flex-start",
                        justifyContent: "flex-start",
                        overflowY: "scroll",
                        maxHeight: "20vh",
                        width: "100%",
                        flexWrap: "wrap",
                    }}
                >
                    {chunk.map((image,j) => <div style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                    }} key={'chunk_div'+j}>
                        <img src={image} alt={classes[analysisResult.selectedClasses[i]]} style={{
                            width: "100px",
                            height: "100px",
                        }} />
                        <span style={{
                            color: analysisResult.selectedClasses[i] === analysisResult.predictions[i*analysisResult.examplePerClass+j] ? "green" : "red",
                        }}>
                            {classes[analysisResult.predictions[i*analysisResult.examplePerClass+j]]}
                        </span>
                    </div>)}
                </div>
            </div>)}
        </div>:null}
    </div>
    {/* <ProgressModal show={isProcessing}>
        <p>Running Model</p>
        <p>{processingMessage}</p>
    </ProgressModal> */}
    
    <Modal show={isProcessing} centered>
        <Modal.Header>
            <Modal.Title>Running Model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <Spinner animation="border" role="status">
                    {/* <span className="sr-only">Loading...</span> */}
                </Spinner>
                <p>{processingMessage}</p>
            </div>
        </Modal.Body>
    </Modal>
    

    </>
}

export default Controls
            
