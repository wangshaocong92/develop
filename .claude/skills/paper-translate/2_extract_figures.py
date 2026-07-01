#!/usr/bin/env python3
"""从 PDF 中提取所有含 Figure 标题的页面，渲染整页并自动裁剪图片。
用法: python3 2_extract_figures.py <pdf_path> <output_dir> [--pages 5,12,13]
     不带 --pages 则自动扫描全部页面。
"""
import sys, os, re
import pypdfium2 as pdfium
from PIL import Image

SCALE = 2
PAGE_W, PAGE_H = 595, 842
MARGIN_X = 48

def find_captions(textpage):
    caps = []
    searcher = textpage.search('Figure')
    result = searcher.get_next()
    while result:
        idx, _ = result
        try:
            bbox = textpage.get_charbox(idx)
        except:
            result = searcher.get_next(); continue
        if bbox[0] < 100:
            try:
                ctx = textpage.get_text_range(index=idx, count=30)
                m = re.match(r'Figure\s+(\d+)', ctx)
                if m:
                    caps.append({'num': int(m.group(1)), 'y': bbox[1], 'x': bbox[0]})
            except: pass
        result = searcher.get_next()
    return caps

def render_page(pdf, page_num, output_dir):
    page = pdf[page_num - 1]
    bitmap = page.render(scale=SCALE)
    img = bitmap.to_pil()
    path = os.path.join(output_dir, f'page{page_num}_full.png')
    img.save(path, 'PNG')
    print(f"  整页渲染: {path} ({img.size})")
    return img

def crop_figure(img, textpage, fig_num, caption_y, output_dir):
    w, h = img.size
    y0_pts = 52
    y1_pts = caption_y - 8
    x0 = int(MARGIN_X * SCALE)
    x1 = int((PAGE_W - MARGIN_X) * SCALE)
    y0 = max(0, int(y0_pts * SCALE))
    y1 = min(h, int(y1_pts * SCALE))
    crop = img.crop((x0, y0, x1, y1))
    path = os.path.join(output_dir, f'fig{fig_num}.png')
    crop.save(path, 'PNG')
    print(f"  裁剪 fig{fig_num}: y={y0_pts}-{y1_pts}pts -> {crop.size}")

def main(pdf_path, output_dir, pages=None):
    os.makedirs(output_dir, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)

    if pages is None:
        print(f"扫描全部 {total_pages} 页中的 Figure 标题...")
        pages = []
        for pg in range(1, total_pages + 1):
            page = pdf[pg - 1]
            textpage = page.get_textpage()
            captions = find_captions(textpage)
            textpage.close()
            if captions:
                pages.append(pg)
                print(f"  第 {pg} 页: {len(captions)} 个 Figure 标题")
        if not pages:
            print("未检测到 Figure 标题。请用 --pages 手动指定页码。")
            pdf.close()
            return
        print(f"共检测到 {len(pages)} 个含图页面: {pages}")

    for pg in pages:
        print(f"\n=== Page {pg} ===")
        page = pdf[pg - 1]
        textpage = page.get_textpage()
        captions = find_captions(textpage)
        captions.sort(key=lambda c: c['y'])
        cap_info = [(c['num'], 'y=' + str(int(c['y']))) for c in captions]
        print(f"  标题: {cap_info}")
        img = render_page(pdf, pg, output_dir)
        for cap in captions:
            crop_figure(img, textpage, cap['num'], cap['y'], output_dir)
        textpage.close()

    pdf.close()
    print(f"\n完成！共处理 {len(pages)} 页。使用 crop_tool.html 可视化微调裁剪区域。")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python3 2_extract_figures.py <pdf_path> <output_dir> [--pages 5,12,13]")
        print("不带 --pages 则自动扫描全部页面检测 Figure 标题。")
        sys.exit(1)
    pdf = sys.argv[1]
    out = sys.argv[2]
    pages = None
    if '--pages' in sys.argv:
        idx = sys.argv.index('--pages')
        pages = [int(p) for p in sys.argv[idx+1].split(',')]
    main(pdf, out, pages)
