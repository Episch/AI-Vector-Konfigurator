#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image


EF_CDN_SCHEMA = "https://images.ernstings-family.com/ean/{variant_id}/01.jpg"


@dataclass(frozen=True)
class ProductImageRef:
    variant_id: str
    name: str | None = None
    category_main: str | None = None
    category_layer: str | None = None
    category_type: str | None = None
    cdn_image_url: str | None = None
    mask_url: str | None = None


def build_cdn_image_url(variant_id: str) -> str:
    """
    Fallback gemäß JSON-Note:
    images.ernstings-family.com/ean/{nr}/01.jpg
    """
    return EF_CDN_SCHEMA.format(variant_id=variant_id)


def _http_get_bytes(url: str, *, timeout_s: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": "ernstingsFamilyAI/0.1 (+Pillow)"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except URLError as e:
        raise RuntimeError(f"Bild konnte nicht geladen werden: {url} ({e})") from e


def load_image_from_url(image_url: str) -> Image.Image:
    data = _http_get_bytes(image_url)
    img = Image.open(io.BytesIO(data))
    return img.convert("RGBA")


def load_product_image(product: dict[str, Any] | ProductImageRef) -> Image.Image:
    if isinstance(product, ProductImageRef):
        variant_id = product.variant_id
        image_url = product.cdn_image_url or build_cdn_image_url(variant_id)
    else:
        variant_id = str(product.get("variant_id", "")).strip()
        image_url = (product.get("cdn_image_url") or "").strip() or build_cdn_image_url(variant_id)

    if not variant_id:
        raise ValueError("Produkt ohne variant_id kann nicht geladen werden.")
    return load_image_from_url(image_url)


def generate_mask(image_url: str) -> Image.Image | None:
    """
    Dummy-Platzhalter für zukünftige Segmentierung.

    Später: hier einen Segmentierungs-Service (z.B. SAM / Nanobanana) aufrufen,
    der eine Alpha-Maske (oder Binärmaske) für das Produkt liefert.

    Rückgabe-Konvention (empfohlen):
    - PIL Image im Mode "L" (0..255), gleiche Größe wie das Input-Bild
    - oder None, wenn keine Maske verfügbar ist.
    """
    _ = image_url
    return None


def _layer_rank(category_layer: str | None, category_main: str | None) -> int:
    """
    Kleinster Rank = unten im Composite.

    Gewünschte Reihenfolge:
    - Hose unten
    - Oberteil/Oberlayer mittig
    - Accessoires/Schmuck/Schuhe oben
    """
    layer = (category_layer or "").strip().lower()
    main = (category_main or "").strip().lower()

    if layer == "hose":
        return 10
    if layer in {"oberteil", "oberlayer"}:
        return 20
    if main in {"accessoires", "schmuck", "schuhe"}:
        return 30
    return 25


def _ensure_same_canvas(images: list[Image.Image], *, bg=(0, 0, 0, 0)) -> tuple[list[Image.Image], tuple[int, int]]:
    """
    Vereinheitlicht Canvas-Größe: nimmt max(W), max(H) und zentriert die restlichen Bilder.
    """
    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)
    out: list[Image.Image] = []
    for im in images:
        if im.size == (max_w, max_h):
            out.append(im)
            continue
        canvas = Image.new("RGBA", (max_w, max_h), bg)
        x = (max_w - im.width) // 2
        y = (max_h - im.height) // 2
        canvas.alpha_composite(im, (x, y))
        out.append(canvas)
    return out, (max_w, max_h)


def compose_outfit(product_list: Iterable[dict[str, Any] | ProductImageRef]) -> Image.Image:
    """
    Nimmt Outfit-Produkte (z.B. Bluse, Hose, Tasche) und legt sie in Layer-Reihenfolge übereinander.
    Hose unten, Oberteil/Oberlayer mittig, Accessoire oben.

    Erwartet pro Produkt mindestens:
    - variant_id
    Optional:
    - cdn_image_url
    - category_layer / category_main
    """
    products = list(product_list)
    if len(products) < 2:
        raise ValueError("compose_outfit benötigt mindestens 2 Produkte.")

    # Sortieren nach Layer-Rank (unten -> oben)
    def key(p: dict[str, Any] | ProductImageRef) -> int:
        if isinstance(p, ProductImageRef):
            return _layer_rank(p.category_layer, p.category_main)
        return _layer_rank(p.get("category_layer"), p.get("category_main"))

    products_sorted = sorted(products, key=key)

    images: list[Image.Image] = []
    for p in products_sorted:
        img = load_product_image(p)

        # Optional: Maske anwenden (aktuell Dummy -> None)
        if isinstance(p, ProductImageRef):
            image_url = p.cdn_image_url or build_cdn_image_url(p.variant_id)
        else:
            image_url = (p.get("cdn_image_url") or "").strip() or build_cdn_image_url(str(p.get("variant_id", "")).strip())

        mask = generate_mask(image_url)
        if mask is not None:
            mask_l = mask.convert("L")
            img.putalpha(mask_l)

        images.append(img)

    images, (w, h) = _ensure_same_canvas(images)
    composite = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for im in images:
        composite.alpha_composite(im)
    return composite


def main() -> None:
    parser = argparse.ArgumentParser(description="Bildservice: Outfit-Bilder laden & übereinanderlegen (Pillow)")
    parser.add_argument(
        "--products-json",
        help="JSON-String oder Pfad zu JSON-Datei mit Produktliste (dicts mit variant_id, cdn_image_url, category_layer/main).",
        required=True,
    )
    parser.add_argument("--out", default="outfit.png", help="Output-Datei (PNG empfohlen)")
    args = parser.parse_args()

    # Input kann Datei oder JSON-String sein
    try:
        if args.products_json.strip().startswith("["):
            products = json.loads(args.products_json)
        else:
            with open(args.products_json, "r", encoding="utf-8") as f:
                products = json.load(f)
    except Exception as e:
        raise SystemExit(f"Konnte products-json nicht lesen: {e}")

    if not isinstance(products, list):
        raise SystemExit("products-json muss eine Liste sein.")

    img = compose_outfit(products)
    img.save(args.out)
    print(json.dumps({"saved_to": args.out, "count": len(products)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

