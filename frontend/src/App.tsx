import React from 'react';
import { Route, Routes } from 'react-router-dom';
import About from './components/About';
import MainView from './components/MainView';
import { TourProvider, useTour } from '@reactour/tour'
import { tutorialSteps } from './tutorialSteps';

function App() {
  return <TourProvider steps={tutorialSteps}>
    <Routes>
      <Route path="/" element={<MainView />} />
      <Route path="/about" element={<About />} />
    </Routes>
  </TourProvider>
}

export default App;
