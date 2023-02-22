import React from 'react'
import Form from 'react-bootstrap/Form';
import * as api from '../api'
import Card from 'react-bootstrap/Card';
import { selectCurrentModel, updateThreshold } from '../features/modelSlice';
// import RangeSlider from 'react-bootstrap-range-slider';
import { useAppDispatch, useAppSelector } from '../app/hooks'
import { setAnalysisResult } from '../features/analyzeSlice';


function Controls() {
    const [image, setImage] = React.useState<File | null>(null)
    const [predictedLabel, setPredictedLabel] = React.useState<string>("Submit for Prediction")

    const dispatch = useAppDispatch()
    const currentModel = useAppSelector(selectCurrentModel)

    const [labels, setLabels] = React.useState<string[]>([])

    React.useEffect(() => {
        api.getLabels()
            .then((res) => {
                setLabels(res)
            })
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
        {/* Form to upload image */}
        <Form style={{
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

            {image?<Card className="mb-3">
                <Card.Img variant="top" src={URL.createObjectURL(image)} />
                <Card.Body>
                    <Card.Text>
                        Predicted Label: {predictedLabel}
                    </Card.Text>
                </Card.Body>
            </Card>:null}

            <Form.Group>
                <Form.Control type="submit" />
            </Form.Group>
        </Form>

        {/* Labels for the class */}
        <h5 className="mb-3">Select Labels to Analyze</h5>
        <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            justifyContent: "flex-start",
            overflowY: "scroll",
        }}>
            {labels.map((label, index) => (
                <Form.Check key={index} type="checkbox" label={label} id={`checkbox_${label}`} />
            ))}
        </div>
        <button className="btn btn-primary mt-5" onClick={() => {
            const checkedLabels = Array.from(document.querySelectorAll("input[type=checkbox]:checked")).map((checkbox) => checkbox.id.split("_")[1])
            const labelIndices = checkedLabels.map((label) => labels.indexOf(label))

            api.analyze(labelIndices)
                .then((res) => {
                    dispatch(setAnalysisResult(res))
                })
        }}>Analyze</button>
    </div>
}

export default Controls