"""Preprocessing for News Source Classification (Project B submission).

The backend gives a CSV with a `url` column only. We:
  1. Derive the label from the domain (foxnews.com -> "FoxNews", else "NBC").
  2. Scrape the headline using requests + BeautifulSoup, matching the example
     in the project guideline (§3.4): targeted h1 selectors per outlet.
  3. Fall back to the URL slug if scraping fails — slugs share most of the
     headline's vocabulary, so the TF-IDF model degrades gracefully.

Returns (X, y): a list of cleaned headline strings and a list of label strings.
"""

import re
import unicodedata
import warnings
from typing import List, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter("ignore", InsecureRequestWarning)


_USER_AGENT = "Mozilla/5.0 (compatible; CIS5190ProjectB/1.0)"
_TIMEOUT = 5.0
_WS_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Same normalization as preprocess.ipynb so train/inference text matches."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return _WS_RE.sub(" ", text).strip()


def _label_from_url(url: str) -> str:
    return "FoxNews" if "foxnews" in url.lower() else "NBC"


def _slug_text(url: str) -> str:
    p = urlparse(url)
    parts = [s for s in p.path.split("/") if s]
    if not parts:
        return ""
    slug = parts[-1]
    if slug.endswith(".print"):
        slug = slug[: -len(".print")]
    slug = slug.replace("-", " ").replace("_", " ")
    return _clean_text(slug)


def _scrape_headline(url: str) -> str:
    try:
        response = requests.get(
            url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT, verify=False
        )
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return ""

    # Outlet-specific selectors. Fox example shown in project guideline §3.4.
    candidates = [
        ("h1", {"class": "headline speakable"}),                       # Fox News
        ("h1", {"class": re.compile(r"article-hero-headline__htag")}), # NBC News
    ]
    for tag, attrs in candidates:
        node = soup.find(tag, attrs=attrs)
        if node and node.get_text(strip=True):
            return _clean_text(node.get_text())

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return _clean_text(h1.get_text())
    title = soup.find("title")
    if title and title.get_text(strip=True):
        return _clean_text(title.get_text())
    return ""


def _text_for_url(url: str) -> str:
    headline = _scrape_headline(url)
    if headline:
        return headline
    return _slug_text(url)


def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    url_col = next((c for c in df.columns if c.lower() == "url"), df.columns[0])
    urls = df[url_col].astype(str).tolist()

    X = [_text_for_url(u) for u in urls]
    y = [_label_from_url(u) for u in urls]
    return X, y
