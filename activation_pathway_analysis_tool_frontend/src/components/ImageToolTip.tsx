import React from 'react'
import { Tooltip } from 'react-tooltip'
import * as api from '../api'

function ImageToolTip({ imgs, imgType, imgData }: { imgs: number[], imgType: 'raw' | 'overlay', imgData: { layer?: string, channel?: number } }) {
    const [imgsUrl, setImgsUrl] = React.useState<string[]>([])

    React.useEffect(() => {
        if(imgType === 'raw')
            api.getInputImages(imgs).then(res => {
                setImgsUrl(res)
            })
            
        else if(imgType === 'overlay' && imgData['layer'] && imgData['channel'])
            api.getActivationOverlay(
                imgs,
                imgData['layer'],
                imgData['channel'],
            ).then(res => {
                    setImgsUrl(res)
            })
    }, [imgs])
    
    return <>
        {imgsUrl.length > 0 && <Tooltip style={{
            opacity: 1
        }} closeOnEsc id="image-tooltip">
            <div style={{
                display: "flex",
                flexDirection: "row",
            }}>
                {imgsUrl.map((img,i) =>
                    <img key={i} src={img} />
                )}
            </div>
        </Tooltip>}
    </>
}

export default ImageToolTip