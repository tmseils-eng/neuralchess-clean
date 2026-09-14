# A deliberately small image: the engine's only runtime dependency is NumPy.
FROM python:3.11-slim

LABEL org.opencontainers.image.title="neuralchess" \
      org.opencontainers.image.description="AlphaZero-style chess engine written from scratch" \
      org.opencontainers.image.source="https://github.com/tmseils-eng/neuralchess"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY neuralchess ./neuralchess
COPY web ./web
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Default: serve the browser UI.  Override for training:
#   docker run --rm -v $PWD/runs:/app/runs neuralchess train --config configs/laptop.json
ENTRYPOINT ["python", "-m", "neuralchess"]
CMD ["serve", "--port", "8000"]
