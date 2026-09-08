#!/usr/bin/env python
"""把 docs/2026-09_量化研究方向指导手册.md 渲染成排版干净的 .docx。
用 python-docx。规则(轻 md 子集):
  # / ## / ###  → Heading1/2/3
  其余段落: --- -> 分隔/留白; 行内 **粗** -> 粗体段; "- "/编号 的简单行做 List; else 正文。
用法: /home/zhulei/anaconda3/envs/zhulei_py312/bin/python scripts/md2docx_guide.py
"""
from __future__ import annotations
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "2026-09_量化研究方向指导手册.md"
OUT = REPO / "docs" / "2026-09_量化研究方向指导手册.docx"

ACC = RGBColor(0x1F, 0x4E, 0x79)
GRAY = RGBColor(0x59, 0x59, 0x59)

doc = Document()
# base style fonts
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

def add_runs(p, text, bold_all=False, size=None, color=None, italic=False):
    # split **bold** tokens
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for tok in parts:
        if not tok: continue
        bold = bold_all
        if tok.startswith("**") and tok.endswith("**"):
            tok = tok[2:-2]; bold = True
        r = p.add_run(tok)
        r.bold = bold; r.italic = italic
        if size: r.font.size = Pt(size)
        if color: r.font.color.rgb = color

def para(text, style=None, bold_all=False, size=None, color=None, italic=False):
    p = doc.add_paragraph(style=style)
    add_runs(p, text, bold_all=bold_all, size=size, color=color, italic=italic)
    return p

lines = SRC.read_text(encoding="utf-8").splitlines()
skip_until_heading = False
for raw in lines:
    line = raw.rstrip()
    if not line.strip():
        continue
    if line.startswith("---"):
        # subtle divider space
        continue
    if line.startswith("# "):
        h = doc.add_heading(line[2:].strip(), level=0)
        for r in h.runs: r.font.color.rgb = ACC
        continue
    if line.startswith("## "):
        h = doc.add_heading(line[3:].strip(), level=1)
        for r in h.runs: r.font.color.rgb = ACC
        continue
    if line.startswith("### "):
        h = doc.add_heading(line[4:].strip(), level=2)
        for r in h.runs: r.font.color.rgb = ACC
        continue
    # blockquote style note
    if line.startswith(">"):
        para(line.lstrip("> "), italic=True, color=GRAY)
        continue
    # numbered / unordered
    m = re.match(r"^(\d+)\.\s+(.*)$", line)
    if m:
        para(f"{m.group(1)}. " + m.group(2), style="List Number", size=10.5); continue
    if line.startswith("- ") or line.startswith("* "):
        para(line[2:], style="List Bullet", size=10.5); continue
    # bold-heavy table-ish or emphasis line -> normal paragraph
    # horizontal emphasis separators
    if re.match(r"^[-=*_ ]{40,}$", line):
        continue
    para(line)

doc.save(OUT)
print("saved", OUT)
