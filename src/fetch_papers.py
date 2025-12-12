import requests
import math
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

QUERY = '("Takayasu arteritis" OR "Takayasu\'s arteritis")'
BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

def fetch_page(page, page_size=100):
    params = {
        "query": QUERY,
        "format": "json",
        "pageSize": page_size,
        "page": page
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    first = fetch_page(1)
    hit_count = int(first["hitCount"])
    page_size = 100
    total_pages = math.ceil(hit_count / page_size)

    print(f"Found {hit_count} results, fetching {total_pages} pages...")

    records = []
    def extract(record):
        return {
            "id": record.get("id"),
            "source": record.get("source"),
            "title": record.get("title"),
            "journal": record.get("journalTitle"),
            "year": record.get("pubYear"),
            "abstract": record.get("abstractText"),
            "mesh_terms": ";".join([m.get("descriptorName", "")
                                    for m in record.get("meshHeadingList", {}).get("meshHeading", [])])
                        if record.get("meshHeadingList") else None,
            "affiliations": record.get("affiliation"),
        }

    # page 1
    records.extend([extract(r) for r in first.get("resultList", {}).get("result", [])])

    # remaining pages
    for page in range(2, total_pages + 1):
        print(f"Fetching page {page}/{total_pages}...")
        data = fetch_page(page)
        results = data.get("resultList", {}).get("result", [])
        records.extend([extract(r) for r in results])

    df = pd.DataFrame(records)
    # drop rows without abstract
    df = df.dropna(subset=["abstract"]).reset_index(drop=True)

    out_path = DATA_DIR / "takayasu_europepmc_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")

if __name__ == "__main__":
    main()
