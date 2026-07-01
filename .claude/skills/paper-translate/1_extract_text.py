#!/usr/bin/env python3
"""从 PDF 中提取文本内容，用于后续翻译。
用法: python3 1_extract_text.py <pdf_path> [output_path]
"""
import sys, os
from PyPDF2 import PdfReader

def extract_text(pdf_path, output_path=None):
    reader = PdfReader(pdf_path)
    print(f"总页数: {len(reader.pages)}")

    lines = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            lines.append(f"\n=== Page {i+1} ===\n")
            lines.append(text)

    result = ''.join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"文本已保存到: {output_path}")

    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 1_extract_text.py <pdf_path> [output_path]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    extract_text(pdf, out)
