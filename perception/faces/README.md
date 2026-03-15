# Face Recognition (Embedding-based)

Face recognition uses **embeddings** instead of reference images. No manual setup required.

## How It Works

1. **First time a face is seen** → Embedding is computed and stored → Assigned `user_1`
2. **Same face seen again** → Embedding matched against stored → Returns `user_1`
3. **New person** → No match → New embedding stored → Assigned `user_2`, `user_3`, ...

Embeddings are stored in `face_embeddings.json` (project root). No face images are saved.

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `EMBEDDINGS_PATH` | `face_embeddings.json` | Where to store embeddings |
| `SIMILARITY_THRESHOLD` | `0.55` | Min similarity to match (0.5–0.7 typical) |

## Legacy: Reference Images (FACES_DB_PATH)

The `faces/` folder is no longer used. Face recognition is fully embedding-based.
