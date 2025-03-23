import React from 'react'
import { Tooltip } from 'react-tooltip'
import * as api from '@api'

function SourceImageToolTip({ imgIdx }: { imgIdx: number }) {
    const [imgUrl, setImgUrl] = React.useState<string>()

    React.useEffect(() => {
        api.getInputImages([imgIdx]).then(res => {
            setImgUrl(res[0])
        })
    }, [imgIdx])
    
    return <>
        {imgUrl && <Tooltip style={{
            opacity: 1
        }} closeOnEsc id="source-image-tooltip">
            <div style={{
                display: "flex",
                flexDirection: "row",
            }}>
                <img src={imgUrl} height={200} width={200} />
            </div>
        </Tooltip>}
    </>
}

export default SourceImageToolTip