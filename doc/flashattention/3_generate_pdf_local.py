#!/usr/bin/env python3
"""从翻译后的 Markdown 生成 PDF（含数学公式渲染、图片嵌入、中文字体）。

依赖:
    pip install markdown weasyprint matplotlib pypdfium2 PyPDF2 Pillow
    # weasyprint 还需要系统库 libpango (conda install -c conda-forge pango)
    # 中文字体: 下载 Noto Sans SC 到 ~/.fonts/

用法: python3 3_generate_pdf.py <markdown_path> [--font <ttf_path>]
"""
import sys, os, re, hashlib, io
import markdown
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from weasyprint import HTML

# ---- 配置 ----
BODY_FONT_SIZE = 10.5       # 正文字号 (pt)
INLINE_MATH_SIZE = 10.0     # 行内公式字号 (pt)
DISPLAY_MATH_SIZE = 10.0    # 块级公式字号 (pt)
MATH_DPI = 200              # 公式渲染 DPI
LINE_HEIGHT = 1.65          # 行高
PAGE_SIZE = 'A4'
PAGE_MARGIN = '2cm 2.2cm'


def fix_latex(latex):
    """修复 matplotlib mathtext 不支持的 LaTeX 命令"""
    latex = re.sub(r'\\le(?![a-zA-Z])', r'\\leq', latex)
    latex = re.sub(r'\\ge(?![a-zA-Z])', r'\\geq', latex)
    return latex


def render_math(latex, output_dir, display_mode=True):
    """将 LaTeX 公式渲染为 PNG 图片，返回相对路径"""
    latex_clean = fix_latex(latex.strip())
    if display_mode:
        latex_clean = re.sub(r'\\tag\{\d+\}', '', latex_clean).strip()

    h = hashlib.md5(latex_clean.encode()).hexdigest()[:12]
    fname = f'math_{h}.png'
    fpath = os.path.join(output_dir, fname)

    # 如果已存在则跳过
    if os.path.exists(fpath):
        return f'figures/math/{fname}'

    fontsize = DISPLAY_MATH_SIZE if display_mode else INLINE_MATH_SIZE

    try:
        # 第一步：测量公式尺寸
        fig_m, ax_m = plt.subplots(figsize=(0.01, 0.01), dpi=MATH_DPI)
        text = ax_m.text(0.5, 0.5, f'${latex_clean}$', fontsize=fontsize,
                         ha='center', va='center', transform=ax_m.transAxes)
        ax_m.axis('off'); fig_m.canvas.draw()
        bbox = text.get_window_extent(renderer=fig_m.canvas.get_renderer())
        bbox_in = bbox.transformed(fig_m.dpi_scale_trans.inverted())
        plt.close(fig_m)

        # 第二步：以精确尺寸渲染
        pad = 0.01
        w_in, h_in = bbox_in.width + pad * 2, bbox_in.height + pad * 2

        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=MATH_DPI)
        ax.text(0.5, 0.5, f'${latex_clean}$', fontsize=fontsize,
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        fig.subplots_adjust(0, 0, 1, 1)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=MATH_DPI, bbox_inches='tight',
                    pad_inches=0, facecolor='white', edgecolor='none')
        plt.close(fig); buf.seek(0)

        with open(fpath, 'wb') as f:
            f.write(buf.read())

        return f'figures/math/{fname}'
    except Exception as e:
        print(f"  [WARN] 公式渲染失败: {str(e)[:60]}")
        return None


def markdown_to_pdf(md_path, pdf_path=None, font_path=None):
    """主函数：Markdown → PDF"""
    base_dir = os.path.dirname(os.path.abspath(md_path))

    if pdf_path is None:
        pdf_path = md_path.replace('.md', '.pdf')

    # 数学公式缓存目录
    math_dir = os.path.join(base_dir, 'figures', 'math')
    os.makedirs(math_dir, exist_ok=True)

    # ---- 1. 读取 Markdown ----
    print("1/5 读取 Markdown...")
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复图片路径为 file:// 绝对路径
    content = content.replace('](figures/', f'](file://{base_dir}/figures/')

    # ---- 2. 渲染数学公式 ----
    print("2/5 渲染数学公式...")
    display_n = len(re.findall(r'\$\$[^$]+\$\$', content))
    inline_n = len(re.findall(r'\$[^$]+\$', content))
    print(f"  块级公式: {display_n}, 行内公式: {inline_n}")

    # 块级公式 $$...$$
    content = re.sub(
        r'\$\$(.+?)\$\$',
        lambda m: f'\n\n![]({render_math(m.group(1), math_dir, True)})\n\n'
                  if render_math(m.group(1), math_dir, True) else m.group(0),
        content, flags=re.DOTALL
    )

    # 行内公式 $...$
    content = re.sub(
        r'\$(.+?)\$',
        lambda m: f'![]({render_math(m.group(1), math_dir, False)})'
                  if render_math(m.group(1), math_dir, False) else m.group(0),
        content
    )

    rendered = len([f for f in os.listdir(math_dir) if f.endswith('.png')])
    print(f"  唯一公式数: {rendered}")

    # ---- 3. Markdown → HTML ----
    print("3/5 转换 HTML...")
    html_body = markdown.markdown(content, extensions=['tables', 'fenced_code'])
    html_body = html_body.replace('src="figures/math/', f'src="file://{base_dir}/figures/math/')

    # ---- 4. 查找中文字体 ----
    if font_path is None:
        for p in [
            os.path.expanduser('~/.fonts/NotoSansSC.ttf'),
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf',
        ]:
            if os.path.exists(p):
                font_path = p
                break

    if font_path is None or not os.path.exists(font_path):
        print("  [WARN] 未找到中文字体，PDF 中文可能乱码。")
        print("  下载: https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf")
        print(f"  放置到 ~/.fonts/ 目录")
        font_face = ''
        font_family = 'sans-serif'
    else:
        print(f"  字体: {font_path}")
        font_face = f"@font-face{{font-family:'NS';src:url('file://{font_path}') format('truetype');}}"
        font_family = "'NS',sans-serif"

    # ---- 5. 生成 PDF ----
    print("4/5 生成 PDF...")
    full_html = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><style>
{font_face}
@page{{size:{PAGE_SIZE};margin:{PAGE_MARGIN};@bottom-center{{content:counter(page);font-size:10px;color:#888;font-family:{font_family};}}}}
body{{font-family:{font_family};font-size:{BODY_FONT_SIZE}pt;line-height:{LINE_HEIGHT};color:#222;text-align:left;}}
h1{{font-size:19pt;text-align:center;margin-bottom:5px;}}
h2{{font-size:14pt;margin-top:26px;margin-bottom:12px;border-bottom:1px solid #333;padding-bottom:4px;}}
h3{{font-size:12pt;margin-top:20px;margin-bottom:10px;}}
h4{{font-size:11pt;margin-top:16px;margin-bottom:8px;}}
p{{margin:6px 0;}}
img{{max-width:100%;height:auto;display:block;margin:8px auto;}}
img[src*="math"]{{display:inline;vertical-align:middle;margin:0;height:0.95em;}}
p img[src*="math"]:only-child{{display:block;height:auto;margin:8px 0;}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:7pt;}}
th,td{{border:1px solid #666;padding:2px 4px;text-align:center;}}
th{{background:#333;color:#fff;}}
pre{{background:#f5f5f5;padding:10px;font-size:8.5pt;overflow-x:auto;border:1px solid #ddd;white-space:pre-wrap;}}
blockquote{{border-left:3px solid #667eea;padding-left:14px;color:#555;margin:10px 0;}}
hr{{border:none;border-top:1px solid #ddd;margin:18px 0;}}sup{{font-size:0.8em;}}strong{{font-weight:600;}}
</style></head><body>{html_body}</body></html>'''

    HTML(string=full_html).write_pdf(pdf_path)

    # ---- 结果 ----
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    size_kb = os.path.getsize(pdf_path) // 1024
    img_count = 0
    for p in reader.pages:
        try:
            for n in p['/Resources']['/XObject']:
                if p['/Resources']['/XObject'][n].get('/Subtype', '') == '/Image':
                    img_count += 1
        except: pass

    print(f"5/5 完成! {pdf_path}")
    print(f"  页数: {len(reader.pages)}, 大小: {size_kb}KB, 图片: {img_count}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 3_generate_pdf.py <markdown_path> [--font <ttf_path>]")
        print("示例: python3 3_generate_pdf.py paper_zh.md")
        sys.exit(1)

    md = sys.argv[1]
    font = None
    if '--font' in sys.argv:
        idx = sys.argv.index('--font')
        font = sys.argv[idx + 1]

    markdown_to_pdf(md, font_path=font)
