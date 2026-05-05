#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import psycopg

from catalog_vector_db import EMBEDDING_MODEL_NAME, embed_texts, get_connection
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class OutfitItem:
    variant_id: str
    name: str | None
    category_main: str | None
    category_layer: str | None
    category_type: str | None
    formality_score: int | None
    reference_url: str | None
    cdn_image_url: str | None
    mask_url: str | None
    similarity: float | None = None


def _fetch_one_dict(cur: psycopg.Cursor[Any]) -> dict[str, Any] | None:
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d.name for d in cur.description]
    return dict(zip(cols, row))


def get_anchor_product(
    conn: psycopg.Connection,
    anchor_variant_id: str,
    table_name: str = "products",
) -> dict[str, Any]:
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
      embedding
    FROM {table_name}
    WHERE variant_id = %(id)s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"id": anchor_variant_id})
        d = _fetch_one_dict(cur)
        if not d:
            raise ValueError(f"Anker-Produkt nicht gefunden: {anchor_variant_id}")
        return d


def _build_where(
    *,
    exclude_ids: Sequence[str],
    category_layer_in: Sequence[str] | None = None,
    category_main_in: Sequence[str] | None = None,
    min_formality: int | None = None,
    max_formality: int | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if exclude_ids:
        clauses.append("variant_id <> ALL(%(exclude_ids)s)")
        params["exclude_ids"] = list(exclude_ids)

    if category_layer_in:
        clauses.append("category_layer = ANY(%(layer_in)s)")
        params["layer_in"] = list(category_layer_in)

    if category_main_in:
        clauses.append("category_main = ANY(%(main_in)s)")
        params["main_in"] = list(category_main_in)

    if min_formality is not None:
        clauses.append("formality_score >= %(min_formality)s")
        params["min_formality"] = int(min_formality)

    if max_formality is not None:
        clauses.append("formality_score <= %(max_formality)s")
        params["max_formality"] = int(max_formality)

    if clauses:
        return "WHERE " + " AND ".join(clauses), params
    return "", params


def vector_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    *,
    top_k: int = 10,
    table_name: str = "products",
    exclude_ids: Sequence[str] = (),
    category_layer_in: Sequence[str] | None = None,
    category_main_in: Sequence[str] | None = None,
    min_formality: int | None = None,
    max_formality: int | None = None,
) -> list[dict[str, Any]]:
    where_sql, where_params = _build_where(
        exclude_ids=exclude_ids,
        category_layer_in=category_layer_in,
        category_main_in=category_main_in,
        min_formality=min_formality,
        max_formality=max_formality,
    )

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
    {where_sql}
    ORDER BY embedding <=> %(q)s
    LIMIT %(k)s;
    """
    params = {"q": query_vector, "k": int(top_k), **where_params}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _first(items: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for x in items:
        return x
    return None


def build_outfit(
    conn: psycopg.Connection,
    anchor_variant_id: str,
    *,
    table_name: str = "products",
    top_k_per_slot: int = 15,
) -> list[OutfitItem]:
    """
    Gibt eine Liste von 3-4 Teilen zurück, die ein Outfit ergeben.
    Enthält immer den Anker als erstes Element.
    """
    anchor = get_anchor_product(conn, anchor_variant_id, table_name=table_name)
    anchor_vec = anchor["embedding"]
    anchor_layer = anchor.get("category_layer")
    anchor_main = anchor.get("category_main")
    anchor_formality = anchor.get("formality_score")

    min_formality = max(1, int(anchor_formality) - 1) if anchor_formality is not None else None
    max_formality = min(5, int(anchor_formality) + 1) if anchor_formality is not None else None

    picked: list[dict[str, Any]] = []
    exclude_ids: list[str] = [anchor_variant_id]

    def pick_one(
        *,
        category_layer_in: Sequence[str] | None = None,
        category_main_in: Sequence[str] | None = None,
    ) -> dict[str, Any] | None:
        candidates = vector_search(
            conn,
            anchor_vec,
            top_k=top_k_per_slot,
            table_name=table_name,
            exclude_ids=exclude_ids,
            category_layer_in=category_layer_in,
            category_main_in=category_main_in,
            min_formality=min_formality,
            max_formality=max_formality,
        )
        chosen = _first(candidates)
        if chosen is None:
            return None
        exclude_ids.append(chosen["variant_id"])
        picked.append(chosen)
        return chosen

    # Regel-Check (minimal, erweiterbar):
    # Wenn der Anker ein (Top-)Teil ist, suchen wir gezielt: Hose + Accessoires
    is_top = anchor_layer in {"Oberteil", "Oberlayer"} or (anchor_main or "").lower().startswith("damen-")

    if is_top:
        pick_one(category_layer_in=["Hose"])

        # 1 Accessoire (Tasche/Gürtel/Tuch/Hut) + 1 Schmuck oder Schuhe, wenn möglich
        pick_one(category_main_in=["Accessoires"])

        second = pick_one(category_main_in=["Schmuck"])
        if second is None:
            pick_one(category_main_in=["Schuhe"])
    else:
        # Fallback: wenn der Anker kein Top ist, ergänze mit einem Top und 1-2 Accessoires.
        pick_one(category_layer_in=["Oberteil", "Oberlayer"])
        pick_one(category_main_in=["Accessoires"])
        pick_one(category_main_in=["Schmuck"])

    # Maximal 3 Ergänzungen (mit Anker = 4 Teile)
    picked = picked[:3]

    def to_item(d: dict[str, Any], *, similarity: float | None = None) -> OutfitItem:
        return OutfitItem(
            variant_id=d["variant_id"],
            name=d.get("name"),
            category_main=d.get("category_main"),
            category_layer=d.get("category_layer"),
            category_type=d.get("category_type"),
            formality_score=d.get("formality_score"),
            reference_url=d.get("reference_url"),
            cdn_image_url=d.get("cdn_image_url"),
            mask_url=d.get("mask_url"),
            similarity=similarity if similarity is not None else d.get("similarity"),
        )

    outfit: list[OutfitItem] = [to_item(anchor, similarity=1.0)]
    outfit.extend(to_item(x) for x in picked)

    # Ziel: 3-4 Items (inkl. Anker). Wenn wir nur 2 Ergänzungen finden, bleibt es bei 3 Items.
    return outfit


def main() -> None:
    parser = argparse.ArgumentParser(description="Outfit-Vorschläge aus pgvector anhand eines Anker-Produkts")
    parser.add_argument("--anchor", required=True, help="variant_id des Anker-Produkts")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"),
        help="PostgreSQL DSN (oder env PG_DSN)",
    )
    parser.add_argument("--table", default="products", help="Tabellenname")
    args = parser.parse_args()

    conn = get_connection(args.dsn)
    try:
        outfit = build_outfit(conn, args.anchor, table_name=args.table)
        print(json.dumps([o.__dict__ for o in outfit], ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

