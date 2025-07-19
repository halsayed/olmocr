import argparse
import json
import os
from typing import List

from openai import OpenAI
from pypdf import PdfReader

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import (
    PageResponse,
    build_finetuning_prompt,
    openai_response_format_schema,
)


def process_page(client: OpenAI, pdf_path: str, page_num: int, target_image_dim: int, target_anchor_len: int) -> str:
    """Render a page and send it to the OpenAI API."""
    image_b64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=target_image_dim)
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=target_anchor_len)

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": build_finetuning_prompt(anchor_text)},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=3000,
        response_format=openai_response_format_schema(),
    )

    result = json.loads(response.choices[0].message.content)
    page = PageResponse(**result)
    return page.natural_text or ""


def process_document(pdf_path: str, target_image_dim: int, target_anchor_len: int) -> List[str]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    reader = PdfReader(pdf_path)
    texts = []
    for i in range(len(reader.pages)):
        texts.append(process_page(client, pdf_path, i + 1, target_image_dim, target_anchor_len))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple OpenAI OCR pipeline")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--output", help="Optional path for saving text")
    parser.add_argument("--target_longest_image_dim", type=int, default=1024)
    parser.add_argument("--target_anchor_text_len", type=int, default=6000)
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY environment variable must be set")

    texts = process_document(args.pdf, args.target_longest_image_dim, args.target_anchor_text_len)
    full_text = "\n".join(texts)

    if args.output:
        with open(args.output, "w") as f:
            f.write(full_text)
    else:
        print(full_text)


if __name__ == "__main__":
    main()
