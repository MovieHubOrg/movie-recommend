from bs4 import BeautifulSoup


def parse_html(html: str) -> str:
    """Parse HTML and extract plain text."""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()


def build_content_string(movie: dict) -> str:
    """Build a content string from movie metadata for embedding generation."""
    title = movie.get("title", "")
    original = movie.get("originalTitle", "")
    desc = parse_html(movie.get("description", ""))
    categories = " ".join(c["name"] for c in movie.get("categories", []))
    country = movie.get("country", "")
    year = str(movie.get("year", ""))

    return f"{title} {original} {desc} {categories} {country} {year}"
