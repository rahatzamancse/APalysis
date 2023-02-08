import React from 'react'
import Card from 'react-bootstrap/Card'

import { useAppSelector } from '../app/hooks'
import { selectCurrentModel } from '../features/modelSlice'

function LayerDetails() {

    const currentModel = useAppSelector(selectCurrentModel)
    const [images, setImages] = React.useState<string[]>([])

    React.useEffect(() => {
        if (currentModel.selectedNode === "") return;
        setImages([])

        Array.from(Array(currentModel.nFilters).keys()).forEach((i) => {
            fetch(`http://127.0.0.1:8000/api/activations/${currentModel.selectedNode}/image/${i}`)
                .then(result => result.blob())
                .then(blob => URL.createObjectURL(blob))
                .then(imgUrl => {
                    setImages(images => [...images, imgUrl])
                })
        })
    }, [currentModel.selectedNode])


    return <div className="rsection" style={{
        display: "flex",
        flexDirection: "column",
        width: "20%",
        minWidth: "300px",
        maxWidth: "500px",
        minHeight: "90vh",
        maxHeight: "90vh",
        overflow: "scroll",
    }}>
        <h2 style={{
            alignSelf: "center",
        }}>{currentModel.selectedNode}</h2>
        <div style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
        }}>
            {images.map((image, i) =>
                <Card className="m-1" key={i} style={{
                    width: "30%"
                }}>
                    <Card.Img variant="top" src={image} style={{
                        opacity: currentModel.imgSummary[i] >= currentModel.threshold ? 1 : 0.2,
                    }}/>
                </Card>
            )}
        </div>

    </div >
}

export default LayerDetails