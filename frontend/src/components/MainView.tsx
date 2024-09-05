import React from 'react'
import GraphViewer from './GraphViewer'
import { ReactFlowProvider } from 'reactflow'

function MainView() {
  return <div style={{
    display: "flex",
    flexDirection: "row",
  }}>
    <ReactFlowProvider>
      <GraphViewer />
    </ReactFlowProvider>
  </div>
}

export default MainView