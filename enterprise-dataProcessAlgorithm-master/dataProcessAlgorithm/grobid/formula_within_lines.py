import json
import os
import re
import time
import fitz
import string


# special letters
LETTERS = string.ascii_letters  # english letters
PUNCTUATION = string.punctuation  # punctuations
OPERATION = list(r"!/=*+|^") + ["×", "÷", "°", "″", "′", "•",
                                "sin", "cos", "tan", "arcsin", "arctan", 'arccos', "log", "ln", "exp", "log2", "log10",
                                "km", "kg", "mol", "pa", "wb", "°C", "rad", "lm", "lx", "hz", "bq", "gy", "sv", "sr", "kn"]  # operators

for s in OPERATION:
    PUNCTUATION = PUNCTUATION.replace(s, "")
GREEK = [chr(code) for code in range(0x0370, 0x0400)]  # greek letters
MATH = [chr(code) for code in range(0x2200, 0x2300)]  # math symbols & "degree", "minute", "second"
FORMULA_CHAR = OPERATION + GREEK + MATH


def inline_math(text: str, pdf_path: str) -> str:
    """Is a math expression within a line"""
    parsed_pdf = parse_pdf(pdf_path)
    parsed_pdf = filter_math(parsed_pdf)     # only the formula preserved
    text_list = split_text_by_coords(text)

    coord_flag = False
    for i in range(len(text_list)):
        if coord_flag:
            pages_index, rects_given, coords_str, coord_num = coords_transfer(text_list[i])
            for coord_index in range(coord_num):    # index in coord group, not whole paper
                rect = rects_given[coord_index]
                coord = coords_str[coord_index]
                try:
                    page = parsed_pdf[pages_index[coord_index]]
                    text_list[i] = match_page(text_list[i], page, rect, coord)
                except IndexError:
                    new_coord = ",".join([coord, "0"])
                    text_list[i] = text_list[i].replace(coord, new_coord)
                    # text_list[i] = ",".join([text_list[i], "0"])

            text_list[i] = "".join(['"', text_list[i], '"', text_list[i+1]])    # if `index out of range` raised

        coord_flag = not coord_flag

    text = [text_list[i] for i in range(len(text_list)) if i % 2 == 1]
    text.insert(0, text_list[0])
    return "coords=".join(text)


def filter_math(pdf_json):
    filtered_pdf_json = []

    for page_id, page in enumerate(pdf_json):
        if page is None:
            continue

        blocks = page.get("blocks", None)
        if blocks is None:
            continue

        filtered_blocks = []

        for block_id, block in enumerate(blocks):
            if block is None:
                continue

            lines = block.get("lines", None)
            if lines is None:
                continue

            filtered_lines = []

            for line_id, line in enumerate(lines):
                if line is None:
                    continue

                spans = line.get("spans", None)
                if spans is None:
                    continue

                filtered_spans = []

                for span_id, span in enumerate(spans):
                    if span is None:
                        continue

                    if is_math(span):
                        filtered_spans.append(span)

                line["spans"] = filtered_spans
                if filtered_spans != []:
                    filtered_lines.append(line)

            block["lines"] = filtered_lines
            if filtered_lines != []:
                filtered_blocks.append(block)

        page["blocks"] = filtered_blocks
        if filtered_blocks != []:
            filtered_pdf_json.append(page)

    return filtered_pdf_json


def match_page(text: str, page: dict, rect_given: list, coord_str: str) -> str:
    blocks = page["blocks"]
    for block in blocks:
        rect_block = block["bbox"]
        if not is_cover(rect_block, rect_given):
            continue
        try:
            lines = block["lines"]
        except KeyError:
            continue
        for line in lines:
            rect_line = line["bbox"]
            if not is_cover(rect_line, rect_given):
                continue

            spans = line["spans"]
            for span in spans:
                rect_span = span["bbox"]
                flag_cover = is_cover(rect_given, rect_span)
                if not flag_cover:
                    continue

                if flag_cover:
                    flag_math = is_math(span)
                    if flag_math:
                        replacement = ",".join([coord_str, "1"])
                        text = text.replace(coord_str, replacement)
                        return text

    replacement = ",".join([coord_str, "0"])
    text = text.replace(coord_str, replacement)
    return text


def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    texts = []
    for i, page in enumerate(doc.pages()):
        text = page.get_text("dict")
        texts.append(text)
    return texts


def split_text_by_coords(text, pattern=r'coords="([^"]+)"'):
    return re.split(pattern, text)


def coords_transfer(coord_str: str):
    pages = []
    rects_given = []
    coords_str = []
    coords = coord_str.split(";")
    for coord in coords:
        page, x, y, l, h = [float(i) for i in coord.split(",")]
        rect_given = [x, y, x + l, y + h]

        coords_str.append(coord[0:])  # 1,2,3,4,5  '1,' is page index

        pages.append(int(page) - 1)
        rects_given.append(rect_given)
    return pages, rects_given, coords_str, len(coords)


####输入是小框和大框
def bb_intersection_over_small(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    sml_lx = boxA[0]  ###小框左边界
    sml_rx = boxA[2]  ###小框右边界
    sml_uy = boxA[1]  ###小框上边界
    sml_dy = boxA[3]  ###小框下边界


    bg_lx = boxB[0]
    bg_rx = boxB[2]
    bg_uy = boxB[1]
    bg_dy = boxB[3]



    xA = max(sml_lx, bg_lx)##max(boxA[0], boxB[0])###覆盖区域左边界
    yA = max(sml_uy, bg_uy)##(boxA[1], boxB[1])###覆盖区上界
    xB = min(sml_rx, bg_rx)###覆盖区右边界
    yB = min(sml_dy, bg_dy)###覆盖区下边界

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (sml_rx - sml_lx + 1) * (sml_dy - sml_uy + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea)###覆盖区域占boxa比例

    return iou


# ##box坐标表示[x_ld,y_ld, x_ru, y_ru]左下点右上点
def is_cover(largebbox, smallbbox):
    sml_lx = smallbbox[0]###小框左边界
    sml_rx = smallbbox[2]###小框右边界
    sml_uy = smallbbox[1]###小框上边界
    sml_dy = smallbbox[3]###小框下边界

    bg_lx = largebbox[0]
    bg_rx = largebbox[2]
    bg_uy = largebbox[1]
    bg_dy = largebbox[3]

    # if smallbbox[3] < largebbox[1] or smallbbox[1] > largebbox[3] or smallbbox[0] > largebbox[2] or smallbbox[2] < largebbox[0]:
    #     return False
    if sml_dy < bg_uy or sml_uy > bg_dy or sml_lx > bg_rx or sml_rx < bg_lx:
        return False
    # 小bbox上下与大bbox偏差过大-->不同行-->不包含
    largecentery = (bg_dy+bg_uy) /2
    smallcentery = (sml_uy+sml_dy) /2
    # if abs(largecentery - smallcentery) > (largebbox[3] - largebbox[1])/2:
    if abs(largecentery - smallcentery) > (bg_dy-bg_uy) / 2:
        return False

    # 两个重叠的面积小于一定比例-->不包含
    if bb_intersection_over_small(smallbbox, largebbox)< 0.5:
        return False
    return True###否则包含


def flags_decomposer(flags: int) -> str:
    """Make font flags human readable."""
    """
    flags: 1,2,4,8,......
    return: "superscript, italic, ..."
    """
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)


def is_math(span: dict) -> bool:
    """Is math formula within lines"""
    """
    span: {size: xxx, text: xxx, bbox:xxx, ...}
    return: contains math formula for True, else for False
    """

    font = span["font"]
    flag = span["flags"]
    size = span["size"]
    text = span["text"]

    flag = flags_decomposer(flag)

    # does text contain math, greek, operation characters
    if isin(text, FORMULA_CHAR) or isin(text.split(" "), FORMULA_CHAR):
        return True

    # assuming that formula may be shown in italic
    if isin({"italic"}, font.lower()):
        return True

    # superscript or subscript are usually appeared in formula
    if isin({"super", "sub", "itlic"}, flag):
        return True

    return False


def isin(x, y) -> bool:
    """
    :param x: any instance iterable [x1, x2, x3, ...]
    :param y: any instance iterable [y1, y2, y3, ...]
    :return: is element in x inside y
    """
    for i in x:
        if i in y:
            return True
    return False


if __name__ == "__main__":
    # pdf_root, _, pdf = next(os.walk("/home/hanjiayi/Downloads/inline_math_test/paper"))
    # mmd_root, _, mmd = next(os.walk("/home/hanjiayi/Downloads/inline_math_test/mmd"))
    #
    # pdf = [i for i in pdf if i.startswith("W")][0]
    # mmd = [i for i in mmd if i.startswith("W")][0]

    pdf_list = ['SusDev.15.4%3A10.pdf',
                'Tsunami - The Underrated Hazard 2nd ed - E. Bryant (Springer, 2008) WW.pdf',
                'TAO.2009.05.13.01%28A%29.pdf',
                'AMGH.v48i4%28482%29.pdf',
                '%28asce%29gt.1943-5606.0000298.pdf',
                'Warf - Encyclopedia of Human Geography (Sage, 2006).pdf',
                '21-%283-4%29-3662.pdf',
                "9781316474303.pdf"]
    mmd_list = ['SusDev.15.4%3A10.mmd',
                'Tsunami.mmd',
                'TAO.2009.05.13.01%28A%29.mmd',
                'AMGH.v48i4%28482%29.mmd',
                '%28asce%29gt.1943-5606.0000298.mmd',
                'Warf.mmd',
                '21-%283-4%29-3662.mmd',
                # "20240223_20a26ef5-dd69-43c7-b55b-d566e478c78d.mmd",
                "20240216_20a26ef5-dd69-43c7-b55b-d566e478c78d.mmd"]

    index = -1
    pdf_root, pdf = "/home/hanjiayi/Downloads/inline_math_test/paper", pdf_list[index]
    mmd_root, mmd = "/home/hanjiayi/Downloads/inline_math_test/mmd", mmd_list[index]
    write_path = "/home/hanjiayi/Downloads/inline_math_test/processed/{}-old.txt".format(pdf.split(".")[0])

    # pdf_root, pdf = "/home/hanjiayi/Downloads/lzy_send", "9780080455952.pdf"
    # mmd_root, mmd = "/home/hanjiayi/Downloads/lzy_send", "replaced.mmd"
    # write_path = "/home/hanjiayi/Downloads/lzy_send/9780080455952.mmd"

    with open(os.path.join(mmd_root, mmd), "r", encoding="utf-8") as f:
        text = f.readlines()
    pdf_path = os.path.join(pdf_root, pdf)
    text = "\n".join([i.strip("\n") for i in text])

    s = time.time()
    t = inline_math(text, pdf_path)

    with open(write_path, "w") as f:
        f.writelines(t)

    l = []
    tail = []
    _ = re.findall(r'coords="([^"]+)"', t)
    for i in _:
        for j in i.split(";"):
            a = j.split(",")
            l.append(len(a))
            tail.append(a[-1])
