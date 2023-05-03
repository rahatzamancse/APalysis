import React from 'react'
import { Card, Form } from 'react-bootstrap'
import * as api from '../../api'
import ReactLassoSelect from "react-lasso-select";
import { useAppDispatch } from '../../app/hooks'
import { useAppSelector } from '../../app/hooks';
import { selectFeatureHunt, setUploadComplete } from '../../features/featureHuntSlice';

function ImageSelection() {
    const [image, setImage] = React.useState<string>()
    const [points, setPoints] = React.useState<{ x: number; y: number }[]>([]);
    const [imgChanged, setImgChanged] = React.useState<boolean>(false)
    const dispatch = useAppDispatch()
    const featureHuntState = useAppSelector(selectFeatureHunt)
    
    console.log(featureHuntState)
    
    React.useEffect(() => {
        console.log("fired")
        api.getFeatureHuntImage()
            .then((res) => {
                setImage(res)
            })
    }, [imgChanged])
    
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
        <Form style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            justifyContent: "flex-start",
            width: "100%",
            marginBottom: "20px",
        }} onSubmit={(e) => {
            e.preventDefault()
            api.submitBoxSelection(points)
                .then((res) => {
                    dispatch(setUploadComplete(true))
                })
        }}
        >
            <Form.Group controlId="formFile" className="mb-3">
                <Form.Label>Upload an image</Form.Label>
                <Form.Control name="exampleimg" type="file" accept='image/*'
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        if (e.target.files) {
                            api.submitFeatureHuntImage(e.target.files[0])
                            setImgChanged(val => !val)
                        }
                    }}
                />
            </Form.Group>
            {/* {image ? <Card className="mb-3">
                <Card.Img variant="top" src={URL.createObjectURL(image)} />
                <Card.Body>
                    <Card.Text>
                        Predicted Label: {predictedLabel}
                    </Card.Text>
                </Card.Body>
            </Card> : null} */}
            {image && <><ReactLassoSelect
                value={points}
                src={image}
                imageStyle={{ width: "100%" }}
                onChange={(path) => {
                  setPoints(path);
                }}
                onComplete={(path) => {
                  if (!path.length) return;
                  console.log(path)
                }}
              />
              <div>Points: {points.map(({ x, y }) => `(${x},${y})`).join(" | ")}</div>
            <Form.Group>
                <Form.Control type="submit" />
            </Form.Group></>}
        </Form>
    </div>

}

export default ImageSelection