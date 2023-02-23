import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../app/store';
import * as api from '../api'

export interface AnalysisConfig {
    // selected class indices
    selectedClasses: number[]
    // examples per class
    examplePerClass: number
    // User selected Images indices
    selectedImages: number[]
}

const initialState: AnalysisConfig = {
    selectedClasses: [],
    examplePerClass: 0,
    selectedImages: []
}

export const analysisResultSlice = createSlice({
    name: 'analysisResult',
    initialState,
    reducers: {
        setAnalysisResult: (state, action: PayloadAction<AnalysisConfig>) => {
            state.selectedClasses = action.payload.selectedClasses;
            state.examplePerClass = action.payload.examplePerClass;
        },
        setSelectedImgs: (state, action: PayloadAction<number[]>) => {
            state.selectedImages = action.payload;
        }
    },
});

export const {
    setAnalysisResult,
    setSelectedImgs
} = analysisResultSlice.actions;
export const selectAnalysisResult = (state: RootState) => state.analysisResult;

export default analysisResultSlice.reducer;
