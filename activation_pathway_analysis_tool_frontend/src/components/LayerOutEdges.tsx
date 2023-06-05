import React from 'react'
import { Node } from '../types'

function LayerOutEdges({ node }: { node: Node } ) {
  return (
    <ul>
      <li>{node.out_edge_weight.length}</li>
      {node.out_edge_weight.map(v => <li>{Math.round(v)}</li>)}
    </ul>
  )
}

export default LayerOutEdges