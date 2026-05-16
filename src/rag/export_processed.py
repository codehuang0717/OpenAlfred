"""Export processed markdown for debugging — shows what gets embedded."""
import sys
import asyncio
from pathlib import Path

from rag.md_parser import parse_markdown
from rag.image_handler import process_section_images


async def export(src_path: str, out_path: str, resolve_images: bool = True):
    src = Path(src_path)
    if not src.exists():
        print(f"File not found: {src_path}")
        return

    content = src.read_text(encoding="utf-8", errors="replace")
    sections = parse_markdown(content)
    doc_id = "debug_export"

    output_lines = []
    for sec in sections:
        text = sec.text
        if sec.images:
            text = await process_section_images(text, str(src.parent), doc_id)

        if sec.heading:
            prefix = "#" * sec.heading_level + " " + sec.heading
            output_lines.append(prefix + "\n\n" + text)
        else:
            output_lines.append(text)

    result = "\n\n".join(output_lines)

    if resolve_images:
        # Resolve JSON placeholders back to markdown for readability
        result = _resolve_placeholders(result)

    Path(out_path).write_text(result, encoding="utf-8")
    print(f"Exported: {out_path}")
    print(f"Sections: {len(sections)}, Chars: {len(result)}")


def _resolve_placeholders(text: str) -> str:
    """Resolve {"_img":{...}} back to ![alt](url) markdown."""
    import re
    import json
    from db.rag import get_image_by_id

    async def _resolve():
        nonlocal text
        pattern = re.compile(r'\{"_img":\{"i":(\d+),"d":"([^"]*)"\}\}')

        async def _replace(m):
            img_id = int(m.group(1))
            desc = m.group(2)
            img = await get_image_by_id(img_id)
            if img:
                return f"![{img['alt']}]({img['url']})\n[图片描述: {desc}]"
            return m.group(0)

        # Collect and replace in reverse
        matches = list(pattern.finditer(text))
        result = text
        for m in reversed(matches):
            repl = await _replace(m)
            result = result[:m.start()] + repl + result[m.end():]
        return result

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_resolve())
    else:
        import concurrent.futures
        import threading
        fut = concurrent.futures.Future()
        def _run():
            new_loop = asyncio.new_event_loop()
            try:
                fut.set_result(new_loop.run_until_complete(_resolve()))
            finally:
                new_loop.close()
        threading.Thread(target=_run, daemon=True).start()
        return fut.result(timeout=120)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m rag.export_processed <src_md> [out_path] [--raw]")
        print("  --raw  Don't resolve image placeholders (show JSON)")
        sys.exit(1)

    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "data/processed_export.md"
    resolve = "--raw" not in sys.argv
    asyncio.run(export(src, out, resolve))


if __name__ == "__main__":
    main()
