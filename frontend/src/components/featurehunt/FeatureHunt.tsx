import React from 'react'
import ImageSelection from '@components/featurehunt/ImageSelection'
import HighlightedChannels from '@components/featurehunt/HighlightedChannels'

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