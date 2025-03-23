import React from 'react'
import Controls from '@components/Controls'
import GraphViewer from '@components/GraphViewer'
import RightView from '@components/RightView'
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
    {/* <RightView /> */}
  </div>
}

export default MainView