import React from 'react'
import Form from 'react-bootstrap/Form';
import * as api from '../api'
import Card from 'react-bootstrap/Card';

function Controls() {

    const [image, setImage] = React.useState<File | null>(null)
    const [predictedLabel, setPredictedLabel] = React.useState<string>("Submit for Prediction")

    return <div className="rsection" style={{
        display: "flex",
        flexDirection: "column",
        width: "20%",
        minWidth: "300px",
        maxWidth: "500px",
        minHeight: "90vh",
        padding: "20px",
    }}>
        {/* Upload an image with react-bootstrap */}

        <Form onSubmit={(e) => {
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
    </div>
}

export default Controls