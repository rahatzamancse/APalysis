import React from 'react'
import { Tooltip } from 'react-tooltip'
import * as api from '../api'

function ImageToolTip({ imgs, imgType, imgData }: { imgs: number[], imgType: 'raw' | 'overlay', imgData: { layer?: string, channel?: number } }) {
    const [imgsUrl, setImgsUrl] = React.useState<string[]>([])

    React.useEffect(() => {
        if(imgType === 'raw')
            api.getInputImages(imgs.filter(img => img !== -1)).then(res => {
                setImgsUrl(res)
            })
            
        else if(imgType === 'overlay' && imgData['layer'] && imgData['channel'])
            api.getActivationOverlay(
                imgs,
                imgData['layer'],
                imgData['channel'],
            ).then(img => {
                api.getKernel(imgData['layer']!, imgData['channel']!).then(kernel => {
                    setImgsUrl([kernel, ...img])
                })
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
                    <img key={i} src={img} height={200} width={200} style={{
                        imageRendering: 'pixelated',
                    }} />
                )}
            </div>
        </Tooltip>}
    </>
}

export default ImageToolTip