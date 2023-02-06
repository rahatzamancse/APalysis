import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../app/store';
import * as api from '../api'

export interface CurrentModel {
    value: {
        selectedNode: null|string,
        nFilters: number,
    }
}

const initialState: CurrentModel = {
    value: {
        selectedNode: null,
        nFilters: 0,
    },
}

export const currentModelSlice = createSlice({
    name: 'currentModel',
    initialState,
    reducers: {
        setSelectedNode: (state, action: PayloadAction<CurrentModel["value"]>) => {
            state.value = action.payload;
        },
    },
});

export const {
    setSelectedNode,
} = currentModelSlice.actions;
export const selectCurrentModel = (state: RootState) => state.currentModel.value;

export default currentModelSlice.reducer;
