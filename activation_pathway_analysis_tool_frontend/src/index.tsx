import React from 'react';
import { Provider } from 'react-redux';
import { store } from './app/store';
import ReactDOM from 'react-dom/client';
import App from './App';
import { BrowserRouter as Router } from "react-router-dom"

// Importing the Bootstrap CSS
import 'bootstrap/dist/css/bootstrap.min.css';
import './styles/index.css'
import 'react-tooltip/dist/react-tooltip.css'

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <Router>
      <Provider store={store}>
        <App />
      </Provider>
    </Router>
  </React.StrictMode>
);