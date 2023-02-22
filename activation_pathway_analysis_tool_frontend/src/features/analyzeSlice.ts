import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../app/store';
import * as api from '../api'

export interface AnalysisResult {
    labels: number[],
    coords: [number, number][],
}

const initialState: AnalysisResult = {
    labels: [],
    coords: [],
}

export const analysisResultSlice = createSlice({
    name: 'analysisResult',
    initialState,
    reducers: {
        setAnalysisResult: (state, action: PayloadAction<AnalysisResult>) => {
            state.labels = action.payload.labels;
            state.coords = action.payload.coords;
        },
    },
});

export const {
    setAnalysisResult,
} = analysisResultSlice.actions;
export const selectAnalysisResult = (state: RootState) => state.analysisResult;

export default analysisResultSlice.reducer;
