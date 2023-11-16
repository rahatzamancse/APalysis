# Use a multi-stage build to keep the final image as small as possible

# Stage 1: Build the React app
FROM node:16-alpine as build-stage
WORKDIR /app
COPY activation_pathway_analysis_tool_frontend/package.json ./
RUN npm install
COPY activation_pathway_analysis_tool_frontend/ .
RUN npm run build

# Stage 2: Set up the FastAPI application
FROM python:3.10-slim
WORKDIR /code
COPY activation_pathway_analysis_backend /code
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=build-stage /app/build /code/static

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
