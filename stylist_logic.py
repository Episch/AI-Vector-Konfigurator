#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TypedDict

import psycopg

from catalog_vector_db import EMBEDDING_MODEL_NAME, embed_texts, get_connection
from sentence_transformers import SentenceTransformer


class SeasonSlotConfig(TypedDict, total=False):
    """
    Konfiguration pro Outfit-Slot (ein zu suchendes Teil).
    """

    label: str
    category_layer_in: list[str]
    category_main_in: list[str]
    include_terms: list[str]
    exclude_terms: list[str]
    top_k: int


class SeasonConfig(TypedDict, total=False):
    """
    Gesamtkonfiguration für einen Vibe/Season.
    """

    slots: list[SeasonSlotConfig]
    global_exclude_terms: list[str]


SEASON_CONFIGS: dict[str, SeasonConfig] = {
    "sommer": {
        # Sommer: keine Wolle / langarm-lastigen Teile
        "global_exclude_terms": [
            "wolle",
            "woll",
            "langarm",
            "longarm",
            "winter",
            "fleece",
            "strick",
        ],
        "slots": [
            {
                "label": "Hose (kurz/leicht)",
                "category_layer_in": ["Hose"],
                "include_terms": ["kurz", "leicht", "sommer"],
                "top_k": 25,
            },
            {
                "label": "Oberteil (Top/T-Shirt/leicht)",
                "category_layer_in": ["Oberteil", "Oberlayer"],
                "include_terms": ["top", "t-shirt", "tshirt", "shirt", "leicht", "sommer"],
                "top_k": 25,
            },
            {
                "label": "Hut/Accessoire",
                "category_layer_in": ["Hut"],
                "category_main_in": ["Accessoires"],
                "include_terms": ["strohhut", "hut", "beach", "sommer"],
                "top_k": 25,
            },
        ],
    }
}


def get_outfit_recipe(anchor_category: str | None) -> list[str]:
    """
    Definiert die "Zutaten" für ein vollständiges Outfit/Rendering.

    Rückgabe sind Ziel-Kategorien auf Layer-Ebene (plus spezielle Labels wie "Schuhe"),
    die anschließend jeweils via separater Vektorsuche befüllt werden.
    """
    a = (anchor_category or "").strip().lower()

    # Anker ist Oberteil / Blazer / Oberlayer
    if a in {"oberteil", "oberlayer", "blazer"}:
        return ["Hose", "Tasche", "Schuhe", "Hut"]

    # Anker ist Hose
    if a == "hose":
        return ["Oberteil", "Oberlayer", "Schuhe", "Accessoires"]

    # Fallback: versuche ein generisches Set zu füllen
    return ["Hose", "Oberteil", "Schuhe", "Accessoires"]


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
    keyword_filter: str | None = None,
    include_terms: Sequence[str] | None = None,
    exclude_terms: Sequence[str] | None = None,
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

    patterns: list[str] | None = None
    if keyword_filter:
        kw = str(keyword_filter).strip()
        if kw:
            terms = {
                kw,
                "sommer",
                "leicht",
                "leinen",
                "kurz",
                "sandalen",
            }
            patterns = [f"%{t}%" for t in sorted(terms) if t]

    if patterns:
        clauses.append(
            "(vector_content ILIKE ANY(%(patterns)s) OR attributes::text ILIKE ANY(%(patterns)s))"
        )
        params["patterns"] = patterns

    include_patterns: list[str] | None = None
    if include_terms:
        include_patterns = [f"%{str(t).strip()}%" for t in include_terms if str(t).strip()]
    if include_patterns:
        clauses.append(
            "(vector_content ILIKE ANY(%(include_patterns)s) OR attributes::text ILIKE ANY(%(include_patterns)s))"
        )
        params["include_patterns"] = include_patterns

    exclude_patterns: list[str] | None = None
    if exclude_terms:
        exclude_patterns = [f"%{str(t).strip()}%" for t in exclude_terms if str(t).strip()]
    if exclude_patterns:
        clauses.append(
            "NOT (vector_content ILIKE ANY(%(exclude_patterns)s) OR attributes::text ILIKE ANY(%(exclude_patterns)s))"
        )
        params["exclude_patterns"] = exclude_patterns

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
    keyword_filter: str | None = None,
    include_terms: Sequence[str] | None = None,
    exclude_terms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    where_sql, where_params = _build_where(
        exclude_ids=exclude_ids,
        category_layer_in=category_layer_in,
        category_main_in=category_main_in,
        min_formality=min_formality,
        max_formality=max_formality,
        keyword_filter=keyword_filter,
        include_terms=include_terms,
        exclude_terms=exclude_terms,
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
    keyword_filter: str | None = None,
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

    exclude_ids: list[str] = [anchor_variant_id]

    final_outfit_list: list[dict[str, Any]] = []

    def pick_one(
        *,
        category_layer_in: Sequence[str] | None = None,
        category_main_in: Sequence[str] | None = None,
        include_terms: Sequence[str] | None = None,
        exclude_terms: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any] | None:
        candidates = vector_search(
            conn,
            anchor_vec,
            top_k=int(top_k or top_k_per_slot),
            table_name=table_name,
            exclude_ids=exclude_ids,
            category_layer_in=category_layer_in,
            category_main_in=category_main_in,
            min_formality=min_formality,
            max_formality=max_formality,
            keyword_filter=keyword_filter,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )
        chosen = _first(candidates)
        if chosen is None:
            return None
        exclude_ids.append(chosen["variant_id"])
        final_outfit_list.append(
            {
                "variant_id": chosen.get("variant_id"),
                "name": chosen.get("name"),
                "cdn_image_url": chosen.get("cdn_image_url"),
                "category_layer": chosen.get("category_layer"),
                "category_main": chosen.get("category_main"),
                "category_type": chosen.get("category_type"),
            }
        )
        return chosen

    vibe = (keyword_filter or "").strip().lower()
    season_cfg = SEASON_CONFIGS.get(vibe)

    if season_cfg and season_cfg.get("slots"):
        global_exclude = season_cfg.get("global_exclude_terms", [])

        # Saisonales Set: feste Slots (z.B. Sommer: kurz/leicht, kein Wolle/langarm)
        for slot in season_cfg["slots"]:
            pick_one(
                category_layer_in=slot.get("category_layer_in"),
                category_main_in=slot.get("category_main_in"),
                include_terms=slot.get("include_terms"),
                exclude_terms=[*(slot.get("exclude_terms", []) or []), *global_exclude],
                top_k=slot.get("top_k", top_k_per_slot),
            )
    else:
        # Ensemble-Building (Pflichtenheft): Anker-Layer -> Rezept -> pro fehlendem Layer eigene Suche
        recipe = get_outfit_recipe(anchor_layer)
        for need in recipe:
            n = need.strip().lower()

            # Mapping der "Rezept-Zutaten" auf DB-Filter:
            if n == "hose":
                pick_one(category_layer_in=["Hose"])
            elif n == "oberteil":
                pick_one(category_layer_in=["Oberteil"])
            elif n == "oberlayer":
                pick_one(category_layer_in=["Oberlayer"])
            elif n == "tasche":
                pick_one(category_layer_in=["Tasche"], category_main_in=["Accessoires"])
            elif n == "hut":
                pick_one(category_layer_in=["Hut"], category_main_in=["Accessoires"])
            elif n == "schuhe":
                # In den Daten ist layer oft "Footwear"; main ist "Schuhe"
                chosen = pick_one(category_layer_in=["Footwear"], category_main_in=["Schuhe"])
                if chosen is None:
                    pick_one(category_main_in=["Schuhe"])
            elif n == "accessoires":
                pick_one(category_main_in=["Accessoires"])
            else:
                # unbekannte Zutat -> versuche über layer
                pick_one(category_layer_in=[need])

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
    # `final_outfit_list` ist die minimal benötigte Payload für Rendering-Pipelines,
    # die vollständigen Item-Daten kommen aus den DB-Ergebnissen (oben in pick_one).
    # Für die Rückgabe bauen wir deshalb die vollständigen Items direkt aus den letzten Query-Result-Dicts.
    #
    # Da wir die vollständigen Dicts nicht separat speichern, holen wir sie erneut über variant_id
    # (kleines Dataset, ok) – so bleibt die API stabil.
    for x in final_outfit_list:
        vid = x["variant_id"]
        if not vid:
            continue
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                  variant_id,
                  name,
                  category_main,
                  category_layer,
                  category_type,
                  formality_score,
                  reference_url,
                  cdn_image_url,
                  mask_url
                FROM {table_name}
                WHERE variant_id = %(id)s;
                """,
                {"id": vid},
            )
            full = _fetch_one_dict(cur)
            if full:
                outfit.append(to_item(full))

    # Ziel: 3-4 Items (inkl. Anker). Wenn wir nur 2 Ergänzungen finden, bleibt es bei 3 Items.
    return outfit


def main() -> None:
    parser = argparse.ArgumentParser(description="Outfit-Vorschläge aus pgvector anhand eines Anker-Produkts")
    parser.add_argument("--anchor", required=True, help="variant_id des Anker-Produkts")
    parser.add_argument("--keyword", default=None, help="Optionaler Keyword-Filter (z.B. 'Sommer')")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"),
        help="PostgreSQL DSN (oder env PG_DSN)",
    )
    parser.add_argument("--table", default="products", help="Tabellenname")
    args = parser.parse_args()

    conn = get_connection(args.dsn)
    try:
        outfit = build_outfit(conn, args.anchor, table_name=args.table, keyword_filter=args.keyword)
        print(json.dumps([o.__dict__ for o in outfit], ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

