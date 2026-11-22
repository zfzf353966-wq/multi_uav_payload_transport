#!/usr/bin/env python3
"""将软件使用说明Markdown生成DOCX；DOC/PDF使用LibreOffice转换。"""
from pathlib import Path
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]


def inline(text):
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1（\2）',text)
    return text.replace('**','').replace('`','')


def main():
    doc=Document();sec=doc.sections[0]
    sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=sec.bottom_margin=Cm(1.8);sec.left_margin=sec.right_margin=Cm(2)
    sec.header_distance=sec.footer_distance=Cm(.8)
    normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10.5)
    normal.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.space_after=Pt(4)
    for style,size in [('Title',20),('Heading 1',14),('Heading 2',12)]:
        st=doc.styles[style];st.font.name='Arial';st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string('173B59')
        st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'黑体')
    hp=sec.header.paragraphs[0];hp.text='三无人机城市配送与PPO动态避障仿真系统 · 软件使用说明'
    hp.runs[0].font.size=Pt(9)
    fp=sec.footer.paragraphs[0];fp.alignment=1
    for text,field in [('第 ','PAGE'),(' 页 / 共 ','NUMPAGES')]:
        fp.add_run(text);e=OxmlElement('w:fldSimple');e.set(qn('w:instr'),field);fp._p.append(e)
    fp.add_run(' 页')
    lines=(ROOT/'docs/software_manual.md').read_text().splitlines();i=0;code=False
    while i<len(lines):
        line=lines[i];i+=1
        if line.startswith('```'):
            code=not code;continue
        if code:
            p=doc.add_paragraph();p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.0
            p.paragraph_format.keep_with_next=i<len(lines) and not lines[i].startswith('```')
            run=p.add_run(line or ' ');run.font.name='DejaVu Sans Mono';run.font.size=Pt(8)
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'F2F4F6');p._p.get_or_add_pPr().append(shade)
            continue
        if not line.strip():continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
            rows=[x for x in rows if not re.match(r'^\|[\s:|\-]+\|$',x)]
            table=doc.add_table(rows=0,cols=len(rows[0].strip('|').split('|')));table.style='Table Grid'
            for j,row in enumerate(rows):
                for c,text in zip(table.add_row().cells,row.strip('|').split('|')):
                    c.text=inline(text.strip())
                    for p in c.paragraphs:
                        p.paragraph_format.space_after=Pt(2)
                        for run in p.runs:run.font.size=Pt(9);run.bold=j==0
            continue
        if line.startswith('# '):
            p=doc.add_paragraph(inline(line[2:]),'Title');p.alignment=1
        elif line.startswith('## '):doc.add_paragraph(inline(line[3:]),'Heading 1')
        elif line.startswith('### '):doc.add_paragraph(inline(line[4:]),'Heading 2')
        else:doc.add_paragraph(inline(line))
    path=ROOT/'docs/软件使用说明.docx';doc.save(path);print(path)


if __name__=='__main__':main()
