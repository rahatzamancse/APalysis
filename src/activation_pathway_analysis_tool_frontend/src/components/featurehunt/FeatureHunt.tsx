import React from 'react'
import ImageSelection from './ImageSelection'
import HighlightedChannels from './HighlightedChannels'

function FeatureHunt() {
  return <div style={{
    display: "flex",
    flexDirection: "row",
  }}>
    <ImageSelection />
    <HighlightedChannels />
  </div>
}

export default FeatureHunt