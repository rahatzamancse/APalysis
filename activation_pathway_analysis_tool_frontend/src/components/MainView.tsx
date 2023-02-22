import React from 'react'
import Controls from './Controls'
import GraphViewer from './GraphViewer'
import LayerDetails from './LayerDetails'
import RightView from './RightView'

function MainView() {
  return <div style={{
    display: "flex",
    flexDirection: "row",
  }}>
    <Controls />
    <GraphViewer />
    <RightView />
  </div>
}

export default MainView