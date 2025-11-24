# Graphiti MCP Server with Local Ollama Setup

This guide provides detailed instructions on configuring the Graphiti MCP Server to work with a local Ollama instance and FalkorDB. This setup allows you to run a complete knowledge graph memory system locally, ensuring privacy and zero API costs.

## Prerequisites

1.  **Docker & Docker Compose:** Ensure Docker Desktop or Docker Engine is installed and running.
2.  **Ollama:** Install Ollama from [ollama.com](https://ollama.com).
3.  **Graphiti Repository:** Clone the repository containing the `mcp_server` code.

## Step 1: Prepare Ollama

1.  **Start Ollama:**
    Ensure Ollama is running and accessible. By default, it runs on port 11434.
    ```bash
    ollama serve
    ```

2.  **Pull the Model:**
    Pull the model you intend to use. For this guide, we recommend `gpt-oss:20b` (or `llama3`, `mistral`, etc., depending on your hardware).
    ```bash
    ollama pull gpt-oss:20b
    ```
    *Note: Adjust the model name in the configuration steps below if you choose a different model.*

## Step 2: Configure Environment Variables

Navigate to the `mcp_server` directory and create or edit the `.env` file. This file controls the server's configuration.

```bash
cd mcp_server
cp .env.example .env  # If .env doesn't exist
```

**Edit `.env` with the following configuration:**

```bash
# ============================================
# LLM & OLLAMA CONFIGURATION
# ============================================
# The model name must match the model you pulled in Ollama
MODEL_NAME=gpt-oss:20b

# Base URL for Ollama. 
# IMPORTANT: When running via Docker, use 'http://host.docker.internal:11434/v1' 
# to allow the container to access Ollama on the host machine.
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1

# Dummy key required for the OpenAI client compatibility
OPENAI_API_KEY=ollama

# ============================================
# DATABASE CONFIGURATION (FalkorDB)
# ============================================
# FalkorDB runs in the same Docker network, so we access it via localhost (in the container's view) or service name
FALKORDB_URI=redis://localhost:6379
FALKORDB_DATABASE=default_db
FALKORDB_PASSWORD=

# ============================================
# SYSTEM SETTINGS
# ============================================
# Request Timeout: Local models can be slower, so increase the timeout (in seconds)
REQUEST_TIMEOUT=300

# Logging Level
LOG_LEVEL=INFO

# Concurrency Control
# Controls how many episodes can be processed simultaneously.
# For local setups, keep this low (1-5) to avoid overloading your GPU/CPU.
SEMAPHORE_LIMIT=1
```

## Step 3: Verify `config.yaml`

Ensure your `mcp_server/config/config.yaml` is set up to use the environment variables we just defined. It should look like this:

```yaml
# LLM Configuration - Ollama
llm:
  provider: "openai"                          # We use the OpenAI provider for Ollama compatibility
  model: "${MODEL_NAME:-gpt-oss:20b}"         # Uses env var MODEL_NAME
  temperature: 0.1                            # Lower temperature for more deterministic results
  providers:
    openai:
      api_key: "${OPENAI_API_KEY:-ollama}"    # Uses env var OPENAI_API_KEY
      api_url: "${OLLAMA_BASE_URL:-http://host.docker.internal:11434/v1}"  # Uses env var OLLAMA_BASE_URL

# Embeddings Configuration - Local
embedder:
  provider: "sentence_transformers"           # Uses local Sentence Transformers
  model: "all-MiniLM-L6-v2"                  # Fast, lightweight model included in the image

# Database Configuration - FalkorDB
database:
  provider: "falkordb"
  providers:
    falkordb:
      uri: "${FALKORDB_URI:-redis://localhost:6379}"
      password: "${FALKORDB_PASSWORD:-}"
      database: "${FALKORDB_DATABASE:-default_db}"

# Server Configuration
server:
  transport: "http"
```

## Step 4: Run with Docker Compose

Build and start the server. This will create a container running both the Graphiti MCP server and FalkorDB.

```bash
docker compose up -d --build
```

## Step 5: Verification

1.  **Check Logs:**
    View the logs to ensure everything started correctly.
    ```bash
    docker logs -f docker-graphiti-falkordb-1
    ```
    You should see messages indicating the server is running on port 8000 and connected to FalkorDB.

2.  **Health Check:**
    ```bash
    curl http://localhost:8000/health
    ```
    Response: `{"status":"healthy","service":"graphiti-mcp"}`

3.  **Test Memory Addition (Optional):**
    You can use a tool like `curl` to test adding a memory (via the MCP protocol), or simply connect your MCP client (like Claude Desktop or Cursor).

## Connecting Clients

### Cursor
Configure Cursor's MCP settings:
```json
{
  "mcpServers": {
    "graphiti-memory": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

### Claude Desktop
Use `mcp-remote` as a bridge (since Claude Desktop doesn't natively support HTTP MCP yet):
```bash
npx -y mcp-remote http://localhost:8000/mcp/
```
Add this command to your `claude_desktop_config.json`.

## Troubleshooting

*   **"Connection refused" to Ollama:** Ensure `OLLAMA_BASE_URL` uses `host.docker.internal` instead of `localhost` if running inside Docker.
*   **Database errors:** Ensure the `FALKORDB_URI` is correct. If using the combined container, `redis://localhost:6379` usually works because the app and DB share the container's network namespace (or `redis://falkordb:6379` if using separate services).
*   **Model not found:** Verify the `MODEL_NAME` in `.env` matches exactly what `ollama list` shows.
