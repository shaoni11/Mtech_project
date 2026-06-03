import csv

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def scrape_html_table(url, table_index=0):
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 is required for scraping. Install with: pip install beautifulsoup4")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-dataset-curation/1.0)"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise ValueError("No HTML tables found on the page")
    if table_index >= len(tables):
        raise IndexError(f"table-index {table_index} out of range; found {len(tables)} tables")

    rows = []
    for tr in tables[table_index].find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if cells:
            rows.append([cell.get_text(" ", strip=True) for cell in cells])
    return rows
