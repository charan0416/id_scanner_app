# We explicitly specify the :latest tag to bypass any old caches.
FROM ollama/ollama:latest

# We only need curl for the simple health check in the entrypoint.
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy ONLY the entrypoint script.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]