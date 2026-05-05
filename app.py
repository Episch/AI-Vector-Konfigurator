#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from catalog_vector_db import get_connection
from image_service import build_cdn_image_url, compose_outfit
from stylist_logic import OutfitItem, build_outfit


CATALOG_PATH = Path("_CATALOG_COMPLETE.json")


@dataclass(frozen=True)
class CatalogItem:
    variant_id: str
    name: str
    category_main: str | None
    category_layer: str | None
    category_type: str | None
    formality_score: int | None
    vector_content: str | None
    reference_url: str | None
    cdn_image_url: str | None
    mask_url: str | None


def _as_int_or_none(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_catalog_items(json_path: str) -> list[CatalogItem]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    catalog = (data or {}).get("catalog", {})
    items: list[CatalogItem] = []
    if not isinstance(catalog, dict):
        return items

    for _bucket, products in catalog.items():
        if not isinstance(products, list):
            continue
        for p in products:
            if not isinstance(p, dict):
                continue
            variant_id = str(p.get("variant_id", "")).strip()
            name = str(p.get("name", "")).strip()
            if not variant_id or not name:
                continue

            category = p.get("category") or {}
            attributes = p.get("attributes") or {}
            image_assets = p.get("image_assets") or {}
            if not isinstance(category, dict):
                category = {}
            if not isinstance(attributes, dict):
                attributes = {}
            if not isinstance(image_assets, dict):
                image_assets = {}

            items.append(
                CatalogItem(
                    variant_id=variant_id,
                    name=name,
                    category_main=category.get("main"),
                    category_layer=category.get("layer"),
                    category_type=category.get("type"),
                    formality_score=_as_int_or_none(attributes.get("formality_score")),
                    vector_content=p.get("vector_content"),
                    reference_url=image_assets.get("reference_url"),
                    cdn_image_url=image_assets.get("cdn_image_url"),
                    mask_url=image_assets.get("mask_url"),
                )
            )

    return items


def _product_card(item: CatalogItem) -> None:
    img_url = item.cdn_image_url or build_cdn_image_url(item.variant_id)
    st.image(img_url, use_container_width=True)
    st.markdown(f"**{item.name}**")
    st.caption(
        " · ".join(
            x
            for x in [
                item.category_layer or "-",
                item.category_type or "-",
                f"Formality: {item.formality_score}" if item.formality_score is not None else "Formality: -",
            ]
            if x
        )
    )


def _outfit_item_to_compose_dict(x: OutfitItem) -> dict[str, Any]:
    return {
        "variant_id": x.variant_id,
        "cdn_image_url": x.cdn_image_url,
        "category_layer": x.category_layer,
        "category_main": x.category_main,
        "name": x.name,
        "mask_url": x.mask_url,
    }


def _nano_banana_prompt(outfit: list[OutfitItem]) -> dict[str, Any]:
    """
    Nur Prompt-Vorbereitung (kein API-Call).
    """
    anchor = outfit[0] if outfit else None
    return {
        "task": "final_outfit_render",
        "style": {
            "background": "clean studio, neutral light gray",
            "lighting": "softbox, evenly lit",
            "shadow": "subtle grounded shadow",
            "output": "single outfit composite, realistic textile texture",
        },
        "constraints": [
            "keep product colors faithful",
            "no extra accessories not present in input",
            "no sandals/flip-flops if outfit is business/formal",
        ],
        "anchor": asdict(anchor) if anchor else None,
        "items": [asdict(x) for x in outfit],
        "layer_order_hint": ["Hose", "Oberteil/Oberlayer", "Accessoires/Schmuck/Schuhe"],
        "notes": "mask_url ist aktuell null; später per Segmentierungsservice ergänzen.",
    }


def main() -> None:
    st.set_page_config(page_title="Ernsting's Family – KI Stylist", layout="wide")
    st.title("Ernsting's Family – KI Stylist (Demo)")

    if not CATALOG_PATH.exists():
        st.error("`_CATALOG_COMPLETE.json` nicht gefunden. Bitte ins Projekt-Root legen.")
        st.stop()

    items = load_catalog_items(str(CATALOG_PATH))
    if not items:
        st.error("Katalog enthält keine Produkte (oder Format unerwartet).")
        st.stop()

    with st.sidebar:
        st.header("Datenbank")
        dsn = st.text_input(
            "PG_DSN",
            value=os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"),
            help="Muss auf eine PostgreSQL-DB mit pgvector zeigen, die bereits mit `catalog_vector_db.py` befüllt wurde.",
        )
        table = st.text_input("Tabelle", value="products")
        st.divider()
        st.header("Hinweis")
        st.write("Outfits werden aus der Vektor-DB berechnet. Falls die DB leer ist, zuerst `catalog_vector_db.py` ausführen.")

    st.subheader("Galerie")
    st.write("Klicke ein Produkt an, um ein Outfit zu generieren.")

    if "selected_variant_id" not in st.session_state:
        st.session_state.selected_variant_id = None
    if "outfit" not in st.session_state:
        st.session_state.outfit = None
    if "nano_prompt" not in st.session_state:
        st.session_state.nano_prompt = None

    cols = st.columns(4)
    for idx, item in enumerate(items):
        with cols[idx % 4]:
            _product_card(item)
            if st.button("Outfit generieren", key=f"pick-{item.variant_id}"):
                st.session_state.selected_variant_id = item.variant_id
                st.session_state.outfit = None
                st.session_state.nano_prompt = None

    st.divider()
    st.subheader("Outfit")

    anchor_id = st.session_state.selected_variant_id
    if not anchor_id:
        st.info("Wähle oben ein Anker-Produkt aus.")
        st.stop()

    st.write(f"Anker: `{anchor_id}`")

    if st.session_state.outfit is None:
        with st.spinner("Berechne Outfit-Vorschläge aus der Vektor-DB..."):
            try:
                conn = get_connection(dsn)
                try:
                    outfit = build_outfit(conn, anchor_id, table_name=table)
                finally:
                    conn.close()
                st.session_state.outfit = outfit
            except Exception as e:
                st.error(
                    "Outfit konnte nicht generiert werden. "
                    "Stimmt `PG_DSN` und ist die DB mit `catalog_vector_db.py` befüllt?\n\n"
                    f"Fehler: {e}"
                )
                st.stop()

    outfit: list[OutfitItem] = st.session_state.outfit
    if not outfit:
        st.warning("Kein Outfit gefunden.")
        st.stop()

    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Vorgeschlagene Teile")
        for x in outfit:
            st.write(
                f"- **{x.name or x.variant_id}** "
                f"({x.category_layer or '-'} / {x.category_type or '-'}, "
                f"Formality: {x.formality_score if x.formality_score is not None else '-'}, "
                f"Similarity: {x.similarity:.3f}" if x.similarity is not None else ""
            )
            if x.cdn_image_url:
                st.image(x.cdn_image_url, width=160)

    with right:
        st.markdown("### Outfit-Bild (gestapelt)")
        try:
            composite = compose_outfit([_outfit_item_to_compose_dict(x) for x in outfit])
            st.image(composite, use_container_width=True)
        except Exception as e:
            st.warning(f"Konnte Outfit-Bild nicht rendern: {e}")

    st.divider()
    st.subheader("Finales Rendering vorbereiten")

    if st.button("KI-Styling anwenden"):
        st.session_state.nano_prompt = _nano_banana_prompt(outfit)

    if st.session_state.nano_prompt:
        st.markdown("### Nano-Banana Prompt (Vorbereitung)")
        st.json(st.session_state.nano_prompt)


if __name__ == "__main__":
    main()

