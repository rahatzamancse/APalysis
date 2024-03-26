import React from 'react'
import Form from 'react-bootstrap/Form';
import * as api from '../api'
import { useAppDispatch, useAppSelector } from '../app/hooks'
import { selectAnalysisResult, setAnalysisResult } from '../features/analyzeSlice';
import { chunkify } from '../utils';
import { Modal, Spinner } from 'react-bootstrap';
import { useTour } from '@reactour/tour';
import ModalImage from 'react-modal-image';
import '../styles/control.css'
import '../styles/scrollbar.css'
import { shortenName } from '../utils'


function Controls() {
    const [isProcessing, setIsProcessing] = React.useState<boolean>(false)
    const [processingMessage, setProcessingMessage] = React.useState<string>("")
    const [classes, setClasses] = React.useState<string[]>([])
    const analysisResult = useAppSelector(selectAnalysisResult)
    const [inputImages, setInputImages] = React.useState<string[]>([])
    const [shuffled, setShuffled] = React.useState<boolean>(false)
    const [nExamplePerClass, setNExamplePerClass] = React.useState<number>(5)
    const { setIsOpen } = useTour();

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
                    if (!localStorage.getItem('firstVisit')) {
                        localStorage.setItem('firstVisit', 'true');
                        setTimeout(() => {
                            setIsOpen(true);
                        }, 2000)
                    }
                }
            })
        }, 1000)
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
    
    return <><div className="rsection tutorial-control" style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        width: "23%",
        minWidth: "300px",
        maxWidth: "500px",
        minHeight: "90vh",
        maxHeight: "92vh",
        padding: "8px",
    }}>
        <Form style={{ width: "100%" }}>
            <button title='Reload the last loaded state of the Neural Network' type="button" className="btn btn-primary" style={{
                width: "100%",
                marginBottom: "20px",
                alignSelf: "flex-end"
            }} onClick={(e) => {
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
        </Form>
        <div style={{
            border: "1px solid lightgray",
            borderRadius: "20px",
            padding: "12px",
            width: "100%",
        }}>
            <h5 className="mb-3">Select Labels to Analyze</h5>
            <Form.Check type="switch" id="shufflecheck" label="Shuffle" checked={shuffled} onChange={e => setShuffled(e.target.checked)} className='mb-3 tutorial-shuffle' />
            <Form.Control ref={nExamplePerClassRef} id="nExamplePerClass" className="mb-1  tutorial-image-per-class" type="number" min={1} max={50} placeholder='# Image Per Class' />
            {/* <Form.Label htmlFor="nExamplePerClass" style={{ float: 'left', width: '40%'}}>Image per Class</Form.Label> */}
            <div style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                justifyContent: "flex-start",
                overflowY: "scroll",
                overflowX: "hidden",
                height: "20vh",
                width: "100%",
            }} className='tutorial-selected-classes scrollbar'>
                {classes.map((label, index) => (
                    <Form.Check key={index} type="checkbox" label={label} id={`checkbox_${index}`} ref={(ref: HTMLInputElement) => {checkboxRefs.current[index] = ref}} />
                ))}
            </div>
            <button className="btn btn-primary mt-2 tutorial-analyze"
            style={{
                width: "100%",
            }}
                
            onClick={() => {
                const checkedLabels = checkboxRefs.current.map((checkbox, index) => checkbox.checked).reduce<number[]>((out, bool, index) => bool?out.concat(index):out, [])
                // const checkedLabels = Array.from(document.querySelectorAll("input[type=checkbox]:checked")).map(checkbox => parseInt(checkbox.id.split("_")[1]))
                
                const currentNExamplePerClass = parseInt(nExamplePerClassRef.current?.value || "5")

                api.analyze(checkedLabels, currentNExamplePerClass, shuffled)
                    .then(({ message, task_id }) => {
                        checkTaskStatus(task_id)
                    })
            }}>Analyze</button>
        </div>
        
        <hr />
        <h4 className="mt-2">Input Images</h4>
        {/* Save all images */}
        <button className="btn btn-primary mt-2" style={{
            width: "100%",
        }} onClick={() => {
            api.saveDataset()
        }}>Save All Images</button>
        {inputImages.length > 0 ? <div style={{
            display: "flex",
            paddingLeft: "8px",
            paddingTop: "20px",
            paddingRight: "2px",
            flexDirection: "column",
            marginTop: "20px",
            alignItems: "flex-start",
            justifyContent: "flex-start",
            width: "100%",
            overflowY: "scroll",
            border: "1px solid lightgray",
            borderRadius: "20px",
        }} className='tutorial-input-images scrollbar scrollbar-wide'>
            {chunkify(inputImages, nExamplePerClass).map((chunk, i) => <div key={'outerdiv'+i} style={{
                width: "100%",
                marginBottom: "20px",
            }}>
                { i === 0? <div>
                    <h5>External Images</h5>
                    <div className='scrollbar'
                        key={'div-start'+i}
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
                        <div style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                        }}>
                            <ModalImage hideZoom={false} className='input-image' small={"https://sharkfreewaters.files.wordpress.com/2012/05/shark1.jpg"} large={"https://sharkfreewaters.files.wordpress.com/2012/05/shark1.jpg"} alt={"Adversarial Shark"} />
                            <span style={{ color: "black", }}>
                                {shortenName("trilobite", 10)}
                            </span>
                        </div>
                        <div style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                        }}>
                            <ModalImage hideZoom={false} className='input-image' small={"assets/panda-gibbon-adversarial.png"} large={"assets/panda-gibbon-adversarial.png"} alt={"Adversarial Panda"} />
                            <span style={{ color: "black", }}>
                                {shortenName("trilobite", 10)}
                            </span>
                        </div>
                        <div style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                        }}>
                            <ModalImage hideZoom={false} className='input-image' small={"assets/upload-external.png"} large={"assets/upload-external.png"} alt={"Upload External"} />
                        </div>
                    </div>
                </div>:null}
                <h5 key={'h5'+i}>Class: {classes[analysisResult.selectedClasses[i]]}</h5>
                <div className='scrollbar'
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
                        <ModalImage hideZoom={false} className='input-image' small={image} large={image} alt={classes[analysisResult.selectedClasses[i]]} />
                        <span style={{
                            color: analysisResult.selectedClasses[i] === analysisResult.predictions[i*analysisResult.examplePerClass+j] ? "green" : "red",
                        }}>
                            {classes[analysisResult.predictions[i*analysisResult.examplePerClass+j]] && shortenName(classes[analysisResult.predictions[i*analysisResult.examplePerClass+j]], 10)}
                        </span>
                    </div>)}
                </div>
            </div>)}
        </div>:null}
    </div>
    
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
            
