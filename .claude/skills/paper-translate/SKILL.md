---
name: paper-translate
description: 将英文 PDF 论文翻译为中文 Markdown，提取裁剪图片，生成含公式渲染的 PDF。
---

# 论文翻译与 PDF 生成

## 工作流程

### 步骤 1：提取文本 → 翻译

```bash
python3 .claude/skills/paper-translate/1_extract_text.py paper.pdf paper_text.txt
```

提取 PDF 中的全部文本。人工或借助 LLM 逐段翻译为中文，保存为 Markdown 文件（如 `paper_zh.md`）。

**Markdown 格式要求：**
- 使用 `$...$` 行内公式和 `$$...$$` 块级公式
- 图片引用：`![图 N: 描述](figures/figN.png)`
- 图片下方用 `**图 N | ...**` 添加中文标题
- 表格使用标准 Markdown 表格语法
- **算法/伪代码**使用独立成段格式（见下方算法格式章节）
- **禁止** `$$...$$<br>` 模式——块级公式后不要跟 `<br>`（会导致 KaTeX 解析错误）

#### 算法与伪代码格式

论文中的算法块必须使用特殊格式，**禁止使用 markdown 有序列表**（`1. ` 语法）或 4 空格缩进（会被解析为代码块导致公式无法渲染）：

```markdown
**算法 1：FlashAttention**

**输入：** 矩阵 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ 位于 HBM 中。

1: 设置块大小 $B_c = \lceil M/(4d) \rceil, B_r = \min(\lceil M/(4d) \rceil, d)$。

2: 在 HBM 中初始化 $\mathbf{O} = (0)_{N \times d}, \ell = (0)_N, m = (-\infty)_N$。

5: **for** $1 \leq j \leq T_c$ **do**

6: &emsp;&emsp; 将 $\mathbf{K}_j, \mathbf{V}_j$ 从 HBM 加载到片上 SRAM。

7: &emsp;&emsp; **for** $1 \leq i \leq T_r$ **do**

8: &emsp;&emsp;&emsp;&emsp; 将 $\mathbf{Q}_i, \mathbf{O}_i, \ell_i, m_i$ 从 HBM 加载到片上 SRAM。

14: &emsp;&emsp; **end for**

15: **end for**

16: **返回** $\mathbf{O}$。
```

**要点：**
- 每行独立成段（空行分隔），不用 markdown 列表
- `N:` 编号格式（匹配原论文风格）
- 用 `&emsp;&emsp;`（每级 2 个 em-space）实现缩进，不能用空格（会触发代码块）
- `for...do` / `end for` / `if...then` / `end if` 用 `**粗体**`
- 步骤编号保持不变（不自动重新编号），包括 `for` 行和 `end` 行

### 步骤 2：提取并裁剪图片

**自动检测（推荐）：**

```bash
python3 .claude/skills/paper-translate/2_extract_figures.py paper.pdf figures/
```

自动扫描全部页面，检测 `Figure N` 标题，渲染整页 PNG 并初步裁剪。大部分图片会被自动发现。

**补充遗漏页面：**

部分论文的 Figure 标题不在左边距位置，可能被遗漏。输出日志中会显示检测到的页面列表。若某些图片缺失（如图 3、图 5），手动指定页码补充：

```bash
python3 .claude/skills/paper-translate/2_extract_figures.py paper.pdf figures/ --pages 9,28
```

**微调裁剪区域（重要！）：**

自动裁剪基于固定规则（上方 52pt 到标题上方 8pt），实际效果因论文排版而异。**每次提取后都需要检查裁剪结果**：

1. 将 `crop_tool.html` 复制到 `figures/` 目录：
   ```bash
   cp .claude/skills/paper-translate/crop_tool.html figures/
   ```
2. 在浏览器中打开 `figures/crop_tool.html`
3. 逐张检查：拖拽鼠标框选精确的图片区域
4. 点击「复制坐标」，将结果发给 Claude 以更新裁剪

工具会自动发现 `figures/` 下所有 `page*_full.png` 和 `fig*.png`，无需任何配置。

### 步骤 3：生成 PDF

```bash
python3 .claude/skills/paper-translate/3_generate_pdf.py paper_zh.md
# 如果公式渲染异常，清除缓存后重试：
python3 .claude/skills/paper-translate/3_generate_pdf.py paper_zh.md --clean
```

自动完成：
- LaTeX 数学公式 → PNG 图片（matplotlib）
- 嵌入裁剪后的图表
- 中文字体渲染（需 Noto Sans SC）
- A4 排版，10.5pt 正文
- 自动移除块级公式后的 `<br>` 标签（防止 KaTeX 解析错误）

`--clean` 选项会清除 `figures/math/` 缓存目录，强制重新渲染所有公式。适用于公式渲染结果显示为原始 LaTeX 文本的情况。

## 依赖安装

```bash
# Python 包
pip install PyPDF2 pypdfium2 markdown weasyprint matplotlib Pillow numpy

# weasyprint 系统依赖 (Linux)
conda install -c conda-forge pango

# 中文字体
mkdir -p ~/.fonts
wget -O ~/.fonts/NotoSansSC.ttf \
  https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf
```

## 可调参数

在 `3_generate_pdf.py` 顶部修改：

| 参数 | 默认 | 说明 |
|------|------|------|
| BODY_FONT_SIZE | 10.5 | 正文字号 (pt) |
| INLINE_MATH_SIZE | 10.0 | 行内公式字号 |
| DISPLAY_MATH_SIZE | 10.0 | 块级公式字号 |
| MATH_DPI | 200 | 公式清晰度 |
| LINE_HEIGHT | 1.65 | 行高 |

## 已知限制

- matplotlib mathtext 不支持 `\le` `\ge`，脚本自动替换为 `\leq` `\geq`
- 复杂 LaTeX 公式（`\begin{bmatrix}`、`\begin{cases}` 等矩阵/环境）渲染失败，会保留原始 LaTeX 文本
- 自动裁剪基于固定页边距（左右 48pt），双栏排版等特殊布局需要手动调整
- `crop_tool.html` 需要在 `figures/` 目录中打开才能自动发现图片（或通过工具栏调整扫描范围）
- PDF 使用 weasyprint 流式排版，图片内联显示（无法像 LaTeX 浮动到页眉/页脚）
- 公式缓存偶尔损坏导致部分公式不渲染，使用 `--clean` 清除缓存重试
- 块级公式 `$$...$$` 后紧跟 `<br>` 会导致 KaTeX 解析错误，脚本已自动修复
