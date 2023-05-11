import React from 'react'
import { Node } from '../types'
import * as api from '../api'

function DenseArgmax({ node }: { node: Node } ) {
    const [preds, setPreds] = React.useState<number[]>([])
    
    React.useEffect(() => {
        if(node.layer_type === 'Dense')
            api.getDenseArgmax(node.name).then(res => {
                console.log(res)
                setPreds(res)
            })
    }, [node.name])

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'row',
            flexWrap: 'wrap',
        }}>
            {preds.map((pred, i) =>
                <div key={i} style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    margin: 10,
                }}>
                    <div style={{
                        fontSize: 20,
                        fontWeight: 'bold',
                    }}>
                        {i}
                    </div>
                    <div style={{
                        fontSize: 20,
                        fontWeight: 'bold',
                    }}>
                        {pred}
                    </div>
                </div>
            )}
        </div>
    )
}

export default DenseArgmax