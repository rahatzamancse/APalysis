
import { configureStore, ThunkAction, Action } from '@reduxjs/toolkit';
import currentModelReducer from '../features/modelSlice';

export const store = configureStore({
  reducer: {
    currentModel: currentModelReducer,
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
