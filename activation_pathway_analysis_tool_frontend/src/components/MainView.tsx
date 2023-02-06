import React from 'react'
import Controls from './Controls'
import GraphViewer from './GraphViewer'
import LayerDetails from './LayerDetails'

function MainView() {
  return <div style={{
    display: "flex",
    flexDirection: "row",
  }}>
    <Controls />
    <GraphViewer />
    <LayerDetails />
  </div>
}

export default MainView