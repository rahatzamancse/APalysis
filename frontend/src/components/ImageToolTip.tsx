import React from 'react'
import { Tooltip } from 'react-tooltip'
import * as api from '../api'

function ImageToolTip({ imgs, imgType, imgData, label }: { imgs: number[], imgType: 'raw' | 'overlay', imgData: { layer?: string, channel?: number }, label: string }) {
    const [imgsUrl, setImgsUrl] = React.useState<string[]>([])

    React.useEffect(() => {
    }, [imgs])
    
    return <>
        {imgsUrl.length > 0 && <Tooltip opacity="1" closeOnEsc id="image-tooltip">
            <div style={{
                display: "flex",
                flexDirection: "row",
            }}>
                {imgsUrl.map((img,i) =>
                    <img key={i} src={img} height={200} width={200} alt="" style={{
                        imageRendering: 'pixelated',
                        zIndex: 1000,
                    }} />
                )}
            </div>
            <p style={{
                textAlign: 'center',
                fontSize: '12px',
            }}>{label}</p>
        </Tooltip>}
    </>
}

export default ImageToolTip