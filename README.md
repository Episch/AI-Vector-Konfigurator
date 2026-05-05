# Ernsting's Family Katalog – Embeddings + PostgreSQL (pgvector)

Dieses Mini-Projekt lädt den Produktkatalog aus `_CATALOG_COMPLETE.json`, **flacht** die Daten ab, erzeugt pro Produkt ein **Embedding** (Sentence-Transformers, Modell `all-MiniLM-L6-v2`) und speichert alles in einer **lokalen PostgreSQL-Datenbank mit pgvector**.  
Anschließend kannst du über Vektorsuche ähnliche Produkte finden.

## Voraussetzungen

- **Python 3.10+**
- **PostgreSQL 14+** (empfohlen) mit installierbarer Extension **pgvector**
  - In vielen Setups installierst du pgvector über dein OS-Paketmanagement oder als DB-Extension (abhängig von deiner Distribution).

## Dateien

- `catalog_vector_db.py`: Import/Embedding/DB-Setup + Demo-Query
- `stylist_logic.py`: Outfit-Vorschläge (Regeln + pgvector-Suche)
- `image_service.py`: Bilder laden + (Dummy-)Masken + Outfit-Composite (Pillow)
- `app.py`: Streamlit UI (Galerie → Outfit → Composite → Nano-Banana Prompt)
- `requirements.txt`: Python-Abhängigkeiten
- `_CATALOG_COMPLETE.json`: Katalogdaten (60 Produkte)

## Setup (Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## PostgreSQL Verbindung (DSN)

Das Script nutzt standardmäßig:

- `PG_DSN` aus der Umgebung, oder falls nicht gesetzt:
  - `postgresql://postgres:postgres@localhost:5432/postgres`

Beispiel:

```bash
export PG_DSN="postgresql://postgres:postgres@localhost:5432/postgres"
```

## Ausführen

### Import + Embeddings erzeugen + Upsert in PostgreSQL

```bash
python3 catalog_vector_db.py --json _CATALOG_COMPLETE.json --table products
```

### Tabelle neu aufbauen (Drop + Create)

```bash
python3 catalog_vector_db.py --rebuild --json _CATALOG_COMPLETE.json --table products
```

Beim Run führt das Script am Ende eine kleine Demo-Suche aus und gibt ein JSON mit den Top-Matches aus.

## End-to-End Szenario (Outfit → Composite-Bild → Prompt)

Dieses Beispiel zeigt eine komplette Pipeline mit einem Anker-Produkt (z.B. Blazer), Outfit-Vorschlägen aus der Vektor-DB, lokaler Bild-Komposition und (optional) Prompt-Vorbereitung für ein finales Rendering.

### 1) Setup + DB befüllen

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PG_DSN="postgresql://postgres:postgres@localhost:5432/postgres"
python3 catalog_vector_db.py --rebuild --json _CATALOG_COMPLETE.json --table products
```

### 2) Outfit per `stylist_logic.py` generieren (JSON speichern)

Beispiel-Anker (Blazer): `8664500318`

```bash
python3 stylist_logic.py --anchor 8664500318 --table products > outfit.json
```

`outfit.json` ist eine Liste aus 3–4 Items (Anker + Ergänzungen).

### 3) Produktliste fürs Compositing ableiten und Bild rendern

`image_service.py` erwartet für jedes Item mindestens `variant_id` sowie optional `category_layer`/`category_main` und `cdn_image_url`.

```bash
python3 -c 'import json; o=json.load(open("outfit.json","r",encoding="utf-8")); print(json.dumps([{k:x.get(k) for k in ["variant_id","cdn_image_url","category_layer","category_main","name","mask_url"]} for x in o], ensure_ascii=False))' > outfit_products.json
python3 image_service.py --products-json outfit_products.json --out outfit.png
```

Ergebnis: `outfit.png` (gestapelt: Hose unten → Oberteil/Oberlayer → Accessoires/Schmuck/Schuhe oben).

### 4) Optional: UI nutzen + Nano-Banana Prompt vorbereiten

```bash
streamlit run app.py
```

- In der Galerie ein Produkt auswählen → Outfit wird berechnet.
- „Outfit-Bild (gestapelt)“ wird angezeigt.
- Button **„KI-Styling anwenden“** erzeugt einen JSON-Prompt (nur Vorbereitung), den du später an deinen Nano-Banana Rendering-Service senden kannst.

## Streamlit App

Die App zeigt eine Galerie aller Produkte. Beim Klick auf ein Produkt wird ein Outfit aus der Vektor-DB generiert, die Teile werden angezeigt und ein gestapeltes Outfit-Bild wird gerendert.

Voraussetzung: Die DB ist befüllt (siehe Import oben).

Start:

```bash
streamlit run app.py
```

In der Sidebar kannst du `PG_DSN` und den Tabellennamen setzen.

## Outfit-Vorschläge (Stylist-Logik)

`stylist_logic.py` nimmt ein **Anker-Produkt** (per `variant_id`) und sucht passende Ergänzungen per Vektorsuche in PostgreSQL.

- **Regel-Check**: Wenn der Anker ein `Oberteil`/`Oberlayer` ist, werden gezielt Teile aus `Hose` sowie `Accessoires`/`Schmuck` vorgeschlagen.
- **Formality-Match**: Vorschläge haben maximal **\(\pm 1\)** Abweichung im `formality_score` gegenüber dem Anker.
- **Output**: 3–4 Teile (Anker + bis zu 3 Ergänzungen) als JSON.

Voraussetzung: Die DB ist bereits befüllt (siehe Import oben).

Beispiel:

```bash
python3 stylist_logic.py --anchor 8664500318 --table products
```

## Bild-Service (Pillow)

`image_service.py` lädt die Produktbilder über `cdn_image_url` (oder Fallback-Schema `https://images.ernstings-family.com/ean/{variant_id}/01.jpg`) und kann ein Outfit-Bild als RGBA-Composite erstellen.

- `generate_mask(image_url)`: Dummy-Platzhalter (später Segmentierung via SAM/Nanobanana o.ä.)
- `compose_outfit(product_list)`: stapelt Bilder nach Layer-Reihenfolge:
  - Hose unten
  - Oberteil/Oberlayer mittig
  - Accessoires/Schmuck/Schuhe oben

CLI-Beispiel:

```bash
python3 image_service.py --products-json '[{"variant_id":"8664500318","category_layer":"Oberlayer","category_main":"Damen-Blazer"},{"variant_id":"8611570319","category_layer":"Hose","category_main":"Damen-Hosen"},{"variant_id":"2263930092","category_layer":"Tasche","category_main":"Accessoires"}]' --out outfit.png
```

## DB-Schema (wichtig für spätere Filter)

In der Tabelle (Default: `products`) werden neben dem Embedding u.a. folgende Metadaten als **eigene Spalten** gespeichert:

- `category_layer` (z.B. `"Tasche"`, `"Gürtel"`, `"Oberteil"`, …)
- `formality_score` (1 = sehr casual, 5 = sehr formal)

Zusätzlich werden `attributes`, `category`, `image_assets` und das komplette `raw`-Objekt als `JSONB` gespeichert.

## Ähnlichkeitssuche

Im Script gibt es:

- `find_matching_items(conn, query_vector, top_k=5)`

Die Suche nutzt Cosine-Distanz (`embedding <=> query_vector`) und gibt u.a. `similarity` zurück.

Wenn du Filter (z.B. nur `category_layer='Tasche'` oder `formality_score>=3`) in die Suche integrieren willst, erweitere die SQL-Query in `find_matching_items(...)` um passende `WHERE`-Bedingungen (die Spalten sind bereits vorhanden).

