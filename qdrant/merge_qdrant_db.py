from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

SRC1 = "cu_denver_database"
SRC2 = "cu_denver"
DST  = "cu_denver_merged"

QDRANT_URL = "http://localhost:6333"
BATCH = 256

client = QdrantClient(url=QDRANT_URL)

def ensure_like(src: str, dst: str):
    existing = [c.name for c in client.get_collections().collections]
    if dst in existing:
        return

    info = client.get_collection(src)

    # Handle single-vector collections (most common)
    vectors_cfg = info.config.params.vectors
    # Depending on qdrant-client version, this may already be VectorParams
    size = vectors_cfg.size
    distance = vectors_cfg.distance

    client.create_collection(
        collection_name=dst,
        vectors_config=VectorParams(size=size, distance=distance),
    )

def record_to_point(r):
    """
    r is a qdrant_client.http.models.Record returned by scroll()
    """
    return PointStruct(
        id=r.id,
        vector=r.vector,     # can be list[float] or dict for named vectors
        payload=r.payload,   # dict
    )

def copy_all(src: str, dst: str, batch: int = BATCH):
    offset = None
    total = 0

    while True:
        records, offset = client.scroll(
            collection_name=src,
            limit=batch,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        if not records:
            break

        points = [record_to_point(r) for r in records]
        client.upsert(collection_name=dst, points=points)

        total += len(points)
        print(f"[{src} -> {dst}] copied {total}")

    return total

copy_all(SRC2, DST)

print("Done.")