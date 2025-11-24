# Build stage for frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Production stage
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files from root (not recursively into venv)
COPY api_server.py main.py agent.py orchestrator.py logger.py insights.py ./

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Start the server
CMD ["python", "api_server.py"]
