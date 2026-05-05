# Ernsting's Family Katalog – Embeddings + PostgreSQL (pgvector)

Dieses Mini-Projekt lädt den Produktkatalog aus `_CATALOG_COMPLETE.json`, **flacht** die Daten ab, erzeugt pro Produkt ein **Embedding** (Sentence-Transformers, Modell `all-MiniLM-L6-v2`) und speichert alles in einer **lokalen PostgreSQL-Datenbank mit pgvector**.  
Anschließend kannst du über Vektorsuche ähnliche Produkte finden.

## Voraussetzungen

- **Python 3.10+**
- **PostgreSQL 14+** (empfohlen) mit installierbarer Extension **pgvector**
  - In vielen Setups installierst du pgvector über dein OS-Paketmanagement oder als DB-Extension (abhängig von deiner Distribution).

## Dateien

- `catalog_vector_db.py`: Import/Embedding/DB-Setup + Demo-Query
- `stylist_logic.py`: Outfit-Vorschläge auf Basis eines Anker-Produkts
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

