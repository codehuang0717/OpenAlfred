"""
Markdown section parser — splits .md files by heading boundaries,
extracts image references with alt text and paths.
"""

import re
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("rag.md_parser")

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$", re.MULTILINE)


@dataclass
class ImageRef:
    alt: str
    original_path: str  # relative path from source md, e.g. "./assets/img.png"
    filename: str        # extracted filename, e.g. "img.png"


@dataclass
class Section:
    heading_level: int   # 0 = top-level content before any heading
    heading: str         # heading text, e.g. "Week 2"
    text: str            # body text (image refs preserved)
    images: list[ImageRef] = field(default_factory=list)


def parse_markdown(text: str) -> list[Section]:
    """Parse markdown into heading-delimited sections, extracting image refs.

    Returns a list of Section objects. Top-level content before the first
    heading goes into a section with heading_level=0.
    """
    # Find all heading positions
    heading_matches = list(HEADING_RE.finditer(text))

    if not heading_matches:
        # No headings — treat entire document as one section
        images = list(_extract_images(text))
        clean = _clean_text(text)
        return [Section(heading_level=0, heading="", text=clean, images=images)]

    sections = []
    prev_end = 0

    for i, m in enumerate(heading_matches):
        heading_level = len(m.group(1))
        heading_text = m.group(2).strip()
        body_start = m.end()

        # Body ends at next heading or EOF
        if i + 1 < len(heading_matches):
            body_end = heading_matches[i + 1].start()
        else:
            body_end = len(text)

        body = text[body_start:body_end]

        # Top-level content before first heading
        if i == 0 and m.start() > 0:
            top_text = text[0:m.start()].strip()
            if top_text:
                top_images = list(_extract_images(top_text))
                sections.append(Section(
                    heading_level=0, heading="",
                    text=_clean_text(top_text), images=top_images,
                ))

        images = list(_extract_images(body))
        clean_body = _clean_text(body)
        sections.append(Section(
            heading_level=heading_level,
            heading=heading_text,
            text=clean_body,
            images=images,
        ))
        prev_end = body_end

    logger.debug(
        "Parsed markdown: headings=%d sections=%d total_images=%d",
        len(heading_matches), len(sections),
        sum(len(s.images) for s in sections),
    )
    return sections


def _extract_images(text: str) -> iter:
    for m in IMAGE_RE.finditer(text):
        alt = m.group(1).strip()
        path = m.group(2).strip()
        # Skip external URLs
        if path.startswith(("http://", "https://", "data:")):
            continue
        # Decode URL-encoded path (Chinese chars in paths)
        try:
            from urllib.parse import unquote
            path = unquote(path)
        except Exception:
            pass
        filename = path.replace("\\", "/").rsplit("/", 1)[-1]
        if filename:
            yield ImageRef(alt=alt, original_path=path, filename=filename)


def _clean_text(text: str) -> str:
    """Strip base64 images and excessive whitespace, but preserve markdown image syntax."""
    # Remove base64 encoded images (huge noise)
    text = re.sub(r"!\[[^\]]*\]\(data:image[^)]+\)", "", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
