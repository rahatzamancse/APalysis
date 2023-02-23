import React from 'react'
import Controls from './Controls'
import GraphViewer from './GraphViewer'
import LayerDetails from './LayerActivations'
import RightView from './RightView'
import { ReactFlowProvider } from 'reactflow'

function MainView() {
  return <div style={{
    display: "flex",
    flexDirection: "row",
  }}>
    <Controls />
    <ReactFlowProvider>
      <GraphViewer />
    </ReactFlowProvider>
    <RightView />
  </div>
}

export default MainView