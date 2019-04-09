from pptx import Presentation as ppt
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import os

DEFAULT_LEFT = Inches(0.5)
DEFAULT_TOP = Inches(1.75)
DEFAULT_HEIGHT = Inches(5.5)
DEFAULT_TABLE_LEFT = Inches(3.0)
DEFAULT_TABLE_TOP_TITLE_NEXT = Inches(1.75)
DEFAULT_TABLE_TOP_NEXT = Inches(2.15)
DEFAULT_TABLE_TOP_TITLE_LAST = Inches(4.4)
DEFAULT_TABLE_TOP_LAST = Inches(4.8)
DEFAULT_TABLE_HEIGHT = Inches(2.25)
DEFAULT_TABLE_WIDTH = Inches(7.0)
DEFAULT_FONT = Pt(14)
DEFAULT_TITLE_COLOR = RGBColor(0xFF,0xFF,0xFF)


def add_table_to_slide(title_top, table_top, slide, title, table_matrix):
    try:
        title_shape = slide.shapes.add_textbox(
            left=DEFAULT_TABLE_LEFT,
            top=title_top,
            width=DEFAULT_TABLE_WIDTH,
            height=DEFAULT_TABLE_HEIGHT
        )
        title_shape.text = title
        title_paragraph = title_shape.text_frame.paragraphs[0]
        title_paragraph.font.color.rgb = DEFAULT_TITLE_COLOR
        if table_matrix is None:
            return
        table_shape = slide.shapes.add_table(
            rows=5,
            cols=6,
            left=DEFAULT_TABLE_LEFT,
            top=table_top,
            width=DEFAULT_TABLE_WIDTH,
            height=DEFAULT_TABLE_HEIGHT
        )
        this_table = table_shape.table
        for row in range(len(table_matrix)):
            for col in range(len(table_matrix[row])):
                cell = this_table.cell(row, col)
                cell.text = table_matrix[row][col]
                paragraph = cell.text_frame.paragraphs[0]
                paragraph.font.size = DEFAULT_FONT
                paragraph.alignment = PP_ALIGN.CENTER

    except Exception as e:
        print("WARNING: bad table creation: " + str(e))


class Presentation:
    def __init__(self, savepath, name):
        self.savepath = savepath
        self.name = name
        self.savename = os.path.join(savepath, name)
        self.presentation = ppt()

    def add_image_slide(
            self,
            image_path,
            title='',
            left=DEFAULT_LEFT,
            top=DEFAULT_TOP,
            height=DEFAULT_HEIGHT):
        slide_layout = self.presentation.slide_layouts[5]
        slide = self.presentation.slides.add_slide(slide_layout)
        slide.shapes.add_picture(image_path, left=left, top=top, height=height)
        slide_title = slide.shapes.title
        slide_title.text = title
        return slide

    def image_title(self, t, info):
        title = 'Step ' + str(t)
        if 'competency' in info:
            title += ': ' + info['competency']
        if 'status' in info:
            title += ': ' + info['status']
        if 'package_state' in info:
            title += ': ' + info['package_state']
        return title

    def save(self):
        self.presentation.save(self.savename)

