import React from 'react'
import Form from 'react-bootstrap/Form';
import * as api from '../api'
import Card from 'react-bootstrap/Card';
// import RangeSlider from 'react-bootstrap-range-slider';
import { useAppDispatch, useAppSelector } from '../app/hooks'
import { selectAnalysisResult, setAnalysisResult } from '../features/analyzeSlice';


function Controls() {
    const [uploadOwn, setUploadOwn] = React.useState<boolean>(false)
    const [classes, setClasses] = React.useState<string[]>([])
    const [nExamplePerClass, setNExamplePerClass] = React.useState<number>(5)

    const [inputImages, setInputImages] = React.useState<string[]>([])
    const [inputLabels, setInputLabels] = React.useState<number[]>([])
    
    const checkboxRefs = React.useRef<HTMLInputElement[]>([])

    const dispatch = useAppDispatch()

    React.useEffect(() => {
        api.getLabels().then(setClasses)
    }, [])
    

    return <div className="rsection" style={{
        display: "flex",
        flexDirection: "column",
        width: "20%",
        minWidth: "300px",
        maxWidth: "500px",
        minHeight: "90vh",
        maxHeight: "90vh",
        padding: "20px",
    }}>
        <Form>
            <div className="mb-3">
                <Form.Check
                    type="switch"
                    id="custom-switch"
                    label="Upload your own image"
                    checked={uploadOwn}
                    onChange={e => setUploadOwn(e.target.checked)}
                />
            </div>
            <button type="button" style={{ float: 'right' }} className="btn btn-primary" onClick={(e) => {
                if(checkboxRefs.current.length === 0) return
                api.getConfiguration().then((res) => {
                    setNExamplePerClass(res.examplePerClass)

                    res.selectedClasses.forEach((label, index) => {
                        checkboxRefs.current[index].checked = true
                    })
                    api.getInputImages([...Array(res.selectedClasses.length*res.examplePerClass).keys()]).then(setInputImages)
                    dispatch(setAnalysisResult(res))
                })
            }}><span className="glyphicon glyphicon-refresh">&#xe031;</span></button>
        </Form>
        {uploadOwn ? <h5>Stub Upload own image</h5> : <>
            <h5 className="mb-3">Select Labels to Analyze</h5>
            <Form.Label htmlFor="nExamplePerClass">Image per class</Form.Label>
            <Form.Control id="nExamplePerClass" className="mb-5" type="number" min={1} max={50} value={nExamplePerClass} onChange={e => setNExamplePerClass(parseInt(e.target.value))} />
            <div style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                justifyContent: "flex-start",
                overflowY: "scroll",
                height: "30vh",
            }}>
                {classes.map((label, index) => (
                    <Form.Check key={index} type="checkbox" label={label} id={`checkbox_${index}`} ref={(ref: HTMLInputElement) => {checkboxRefs.current[index] = ref}} />
                ))}
            </div>
            <button className="btn btn-primary mt-5" onClick={() => {
                const checkedLabels = checkboxRefs.current.map((checkbox, index) => checkbox.checked).reduce<number[]>((out, bool, index) => bool?out.concat(index):out, [])
                // const checkedLabels = Array.from(document.querySelectorAll("input[type=checkbox]:checked")).map(checkbox => parseInt(checkbox.id.split("_")[1]))
                console.log("???? ~ checkedLabels:", checkedLabels)

                api.analyze(checkedLabels, nExamplePerClass)
                    .then((res) => {
                        console.log(res)
                        dispatch(setAnalysisResult(res))
                        api.getInputImages([...Array(res.selectedClasses.length*res.examplePerClass).keys()]).then(setInputImages)
                    })
            }}>Analyze</button>
        </>}
        
        {inputImages.length > 0 ? <div>
            <h5 className="mt-5">Input Images</h5>
            <div style={{
                display: "flex",
                flexDirection: "row",
                alignItems: "flex-start",
                justifyContent: "flex-start",
                overflowY: "scroll",
                height: "30vh",
                width: "100%",
                flexWrap: "wrap",
            }}>
                {inputImages.map((image,i) => (
                    <img src={image} key={i} style={{
                        width: "100px",
                        height: "100px",
                        objectFit: "contain",
                    }} />
                ))}
            </div>
        </div>:null}
    </div>


}

export default Controls
            

            /* <Form style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                justifyContent: "flex-start",
                width: "100%",
                marginBottom: "20px",
            }} onSubmit={(e) => {
                e.preventDefault()
                const form = e.currentTarget
                const formElements = form.elements as typeof form.elements & {
                    exampleimg: { files: FileList }
                }
                api.submitImage(form.exampleimg.files[0])
                    .then((res) => {
                        setPredictedLabel(res.prediction)
                    })
            }}
            >
                <Form.Group controlId="formFile" className="mb-3">
                    <Form.Label>Upload an image</Form.Label>
                    <Form.Control name="exampleimg" type="file" accept='image/*'
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            if (e.target.files) {
                                setImage(e.target.files[0])
                            }
                        }}
                    />
                </Form.Group>
                {image ? <Card className="mb-3">
                    <Card.Img variant="top" src={URL.createObjectURL(image)} />
                    <Card.Body>
                        <Card.Text>
                            Predicted Label: {predictedLabel}
                        </Card.Text>
                    </Card.Body>
                </Card> : null}

                <Form.Group>
                    <Form.Control type="submit" />
                </Form.Group>
            </Form> */