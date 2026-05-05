#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@dataclass(frozen=True)
class FlatProduct:
    variant_id: str
    name: str
    category_main: str | None
    category_layer: str | None
    category_type: str | None
    formality_score: int | None
    vector_content: str
    reference_url: str | None
    cdn_image_url: str | None
    mask_url: str | None
    attributes: dict[str, Any]
    category: dict[str, Any]
    image_assets: dict[str, Any]
    raw: dict[str, Any]


def _as_int_or_none(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def load_and_flatten_catalog(json_path: str | Path) -> list[FlatProduct]:
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    catalog = (data or {}).get("catalog", {})
    if not isinstance(catalog, dict):
        raise ValueError("Unerwartetes JSON-Format: `catalog` ist kein Objekt.")

    flat: list[FlatProduct] = []
    for _bucket_name, items in catalog.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue

            variant_id = str(item.get("variant_id", "")).strip()
            name = str(item.get("name", "")).strip()
            vector_content = str(item.get("vector_content", "")).strip()
            if not variant_id or not vector_content:
                continue

            category = item.get("category") or {}
            attributes = item.get("attributes") or {}
            image_assets = item.get("image_assets") or {}
            if not isinstance(category, dict):
                category = {}
            if not isinstance(attributes, dict):
                attributes = {}
            if not isinstance(image_assets, dict):
                image_assets = {}

            flat.append(
                FlatProduct(
                    variant_id=variant_id,
                    name=name,
                    category_main=category.get("main"),
                    category_layer=category.get("layer"),
                    category_type=category.get("type"),
                    formality_score=_as_int_or_none(attributes.get("formality_score")),
                    vector_content=vector_content,
                    reference_url=image_assets.get("reference_url"),
                    cdn_image_url=image_assets.get("cdn_image_url"),
                    mask_url=image_assets.get("mask_url"),
                    attributes=attributes,
                    category=category,
                    image_assets=image_assets,
                    raw=item,
                )
            )

    return flat


def get_connection(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    register_vector(conn)
    return conn


def init_db(conn: psycopg.Connection, table_name: str = "products") -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
              variant_id TEXT PRIMARY KEY,
              name TEXT,
              category_main TEXT,
              category_layer TEXT,
              category_type TEXT,
              formality_score SMALLINT,
              vector_content TEXT NOT NULL,
              reference_url TEXT,
              cdn_image_url TEXT,
              mask_url TEXT,
              embedding vector({EMBEDDING_DIM}) NOT NULL,
              attributes JSONB NOT NULL,
              category JSONB NOT NULL,
              image_assets JSONB NOT NULL,
              raw JSONB NOT NULL,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )

        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {table_name}_layer_idx
              ON {table_name} (category_layer);
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {table_name}_formality_idx
              ON {table_name} (formality_score);
            """
        )

        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw_cos_idx
              ON {table_name}
              USING hnsw (embedding vector_cosine_ops);
            """
        )


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    return [v.tolist() for v in vectors]


def upsert_products(
    conn: psycopg.Connection,
    products: list[FlatProduct],
    embeddings: list[list[float]],
    table_name: str = "products",
) -> None:
    if len(products) != len(embeddings):
        raise ValueError("products und embeddings müssen gleich lang sein.")

    sql = f"""
    INSERT INTO {table_name} (
      variant_id, name, category_main, category_layer, category_type,
      formality_score, vector_content,
      reference_url, cdn_image_url, mask_url,
      embedding, attributes, category, image_assets, raw, updated_at
    )
    VALUES (
      %(variant_id)s, %(name)s, %(category_main)s, %(category_layer)s, %(category_type)s,
      %(formality_score)s, %(vector_content)s,
      %(reference_url)s, %(cdn_image_url)s, %(mask_url)s,
      %(embedding)s, %(attributes)s, %(category)s, %(image_assets)s, %(raw)s, now()
    )
    ON CONFLICT (variant_id) DO UPDATE SET
      name = EXCLUDED.name,
      category_main = EXCLUDED.category_main,
      category_layer = EXCLUDED.category_layer,
      category_type = EXCLUDED.category_type,
      formality_score = EXCLUDED.formality_score,
      vector_content = EXCLUDED.vector_content,
      reference_url = EXCLUDED.reference_url,
      cdn_image_url = EXCLUDED.cdn_image_url,
      mask_url = EXCLUDED.mask_url,
      embedding = EXCLUDED.embedding,
      attributes = EXCLUDED.attributes,
      category = EXCLUDED.category,
      image_assets = EXCLUDED.image_assets,
      raw = EXCLUDED.raw,
      updated_at = now();
    """

    rows: list[dict[str, Any]] = []
    for prod, emb in zip(products, embeddings, strict=True):
        rows.append(
            {
                "variant_id": prod.variant_id,
                "name": prod.name,
                "category_main": prod.category_main,
                "category_layer": prod.category_layer,
                "category_type": prod.category_type,
                "formality_score": prod.formality_score,
                "vector_content": prod.vector_content,
                "reference_url": prod.reference_url,
                "cdn_image_url": prod.cdn_image_url,
                "mask_url": prod.mask_url,
                "embedding": emb,
                "attributes": json.dumps(prod.attributes, ensure_ascii=False),
                "category": json.dumps(prod.category, ensure_ascii=False),
                "image_assets": json.dumps(prod.image_assets, ensure_ascii=False),
                "raw": json.dumps(prod.raw, ensure_ascii=False),
            }
        )

    with conn.cursor() as cur:
        cur.executemany(sql, rows)


def find_matching_items(
    conn: psycopg.Connection,
    query_vector: list[float],
    top_k: int = 5,
    table_name: str = "products",
) -> list[dict[str, Any]]:
    sql = f"""
    SELECT
      variant_id,
      name,
      category_main,
      category_layer,
      category_type,
      formality_score,
      reference_url,
      cdn_image_url,
      mask_url,
      1 - (embedding <=> %(q)s) AS similarity
    FROM {table_name}
    ORDER BY embedding <=> %(q)s
    LIMIT %(k)s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"q": query_vector, "k": int(top_k)})
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _chunked(xs: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main() -> None:
    parser = argparse.ArgumentParser(description="EF-Katalog -> Embeddings -> PostgreSQL/pgvector")
    parser.add_argument("--json", default="_CATALOG_COMPLETE.json", help="Pfad zur Katalog-JSON")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"),
        help="PostgreSQL DSN (oder env PG_DSN)",
    )
    parser.add_argument("--table", default="products", help="Tabellenname")
    parser.add_argument("--rebuild", action="store_true", help="Tabelle droppen und neu erstellen")
    args = parser.parse_args()

    products = load_and_flatten_catalog(args.json)
    if not products:
        raise SystemExit("Keine Produkte gefunden (nach Flattening).")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [p.vector_content for p in products]
    embeddings: list[list[float]] = []
    for batch in _chunked(texts, 64):
        embeddings.extend(embed_texts(model, batch))

    if len(embeddings) != len(products):
        raise SystemExit("Embedding-Anzahl stimmt nicht mit Produktanzahl überein.")

    conn = get_connection(args.dsn)
    try:
        if args.rebuild:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {args.table};")
        init_db(conn, table_name=args.table)
        upsert_products(conn, products, embeddings, table_name=args.table)

        query = "Sommer Boho Tasche Beige"
        q_vec = embed_texts(model, [query])[0]
        matches = find_matching_items(conn, q_vec, top_k=5, table_name=args.table)
        print(json.dumps({"query": query, "top_k": 5, "matches": matches}, ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

