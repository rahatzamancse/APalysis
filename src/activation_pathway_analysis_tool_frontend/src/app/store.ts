
import { configureStore, ThunkAction, Action } from '@reduxjs/toolkit';
import analysisResultReducer from '../features/analyzeSlice';
import featureHuntReducer from '../features/featureHuntSlice';

export const store = configureStore({
  reducer: {
    analysisResult: analysisResultReducer,
    featureHunt: featureHuntReducer,
  },
});

export type AppDispatch = typeof store.dispatch;
export type RootState = ReturnType<typeof store.getState>;
export type AppThunk<ReturnType = void> = ThunkAction<
  ReturnType,
  RootState,
  unknown,
  Action<string>
>;
