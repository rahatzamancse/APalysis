import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../app/store';
import * as api from '../api'

export interface AnalysisResult {
    // Coordinates of the activation pathways using MDS
    coords: [number, number][],
    // GT Label of each image
    labels: number[],
    // User selected images indices
    selectedImgs: number[]
}

const initialState: AnalysisResult = {
    labels: [],
    coords: [],
    selectedImgs: []
}

export const analysisResultSlice = createSlice({
    name: 'analysisResult',
    initialState,
    reducers: {
        setAnalysisResult: (state, action: PayloadAction<AnalysisResult>) => {
            state.labels = action.payload.labels;
            state.coords = action.payload.coords;
        },
        setSelectedImgs: (state, action: PayloadAction<number[]>) => {
            state.selectedImgs = action.payload;
        }
    },
});

export const {
    setAnalysisResult,
    setSelectedImgs
} = analysisResultSlice.actions;
export const selectAnalysisResult = (state: RootState) => state.analysisResult;

export default analysisResultSlice.reducer;
