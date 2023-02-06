import React from 'react';
import Navigation from './components/Navigation';
import { Route, Routes } from 'react-router-dom';
import About from './components/About';
import MainView from './components/MainView';

function App() {
  return <>
    <Navigation />
    <Routes>
      <Route path="/" element={<MainView />} />
      <Route path="/about" element={<About />} />
    </Routes>
  </>;
}

export default App;
