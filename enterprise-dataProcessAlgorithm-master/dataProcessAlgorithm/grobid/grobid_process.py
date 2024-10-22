import json
import os
import logging
import xml.etree.ElementTree as ET
import base64

import paddleocr_process
import utils
from grobid.body import body_process, content_process
from grobid.citation_process import get_references
from xml.dom.minidom import parse
import pandas as pd
import pdfplumber
import oss2
import fasttext
from grobid.formula_within_lines import inline_math
from grobid.relate_coordinate import revise_coodinates_str
from grobid import after_process

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                    datefmt='%Y-%m-%d %A %H:%M:%S',
                    filename='parse_pdf.log',
                    filemode='a')
namespace = {"": 'http://www.tei-c.org/ns/1.0'}


def download_pdf(pdf_url, pdf_id):
    local_path = str(pdf_id) + "/" + str(pdf_id) + ".pdf"
    utils.bucket.get_object_to_file(pdf_url, local_path)
    return


def download_xml(xml_url, pdf_id):
    local_path = str(pdf_id) + "/" + str(pdf_id) + ".grobid.tei.xml"
    utils.bucket.get_object_to_file(xml_url, local_path)
    return


def check_pdf_scanned(file_path, sample_num=8):
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages > sample_num:
                total_pages = sample_num

            image_count = len([1 for i in range(total_pages) if pdf.pages[i].images])
            text_count = len([1 for i in range(total_pages) if pdf.pages[i].extract_text().strip()])
            if text_count >= total_pages - 5:
                return False
            else:
                return True
    except Exception as e:
        logging.error(e)
        return True


def try_get_all_text_from_xml(tree, path, namespace):
    try:
        content = tree.findall(path, namespace)
        return [i.text for i in content]
    except:
        return


def try_get_text_from_xml(tree, path, namespace):
    try:
        content = tree.find(path, namespace).text
        return content
    except:
        return


def get_author(author):
    try:
        content = ""
        for name in author:
            if name.text and content:
                content += " " + name.text
            elif name.text:
                content = name.text
        return content
    except:
        return


def get_meta(namespace, tree):
    title = try_get_text_from_xml(tree, "./teiHeader/fileDesc/titleStmt/title", namespace)
    authors_xml = tree.findall("./teiHeader/fileDesc/sourceDesc/biblStruct/analytic/author/persName", namespace)
    author_list = [get_author(i) for i in authors_xml]
    try:
        language = list(tree.find("./teiHeader", namespace).attrib.values())[0]
    except:
        language = None
    keywords = try_get_all_text_from_xml(tree, "./teiHeader/profileDesc/textClass/keywords/term", namespace)
    published_time = try_get_text_from_xml(tree, "./teiHeader/fileDesc/sourceDesc/biblStruct/monogr/imprint/date",
                                           namespace)
    return {"Title": title,
            "Author": author_list,
            "Language": language,
            "Keywords": keywords,
            "PubDate": published_time}


# 转义成latex文本
def convert_latex_text(text):
    text = text.replace("\\", "\\textbackslash").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#") \
        .replace("}", "\\}").replace("{", "\\{").replace("_", "\\_{}").replace("$", "\\$").replace("~", "\\~{}") \
        .replace("^", "\\^{}").replace("≥", "\\geq").replace("≤", "\\leq")
    return text


def get_table_and_figure(namespace, tree):
    table_list = []
    figure_list = []
    for i in tree.findall("./text/body/figure", namespace):
        namespace_key = ""
        for key in i.attrib.keys():
            if "}id" in key:
                namespace_key = key

        if i.attrib.get("type") == "table" or "tab" in i.attrib.get(namespace_key):
            table_dict = {"head": try_get_text_from_xml(i, "./head", namespace)
            if try_get_text_from_xml(i, "./head", namespace) else "",
                          "label": try_get_text_from_xml(i, "./label", namespace)
                          if try_get_text_from_xml(i, "./label", namespace) else "",
                          "table_title": try_get_all_text_from_xml(i, "./figDesc/div/p/s", namespace)
                          if try_get_all_text_from_xml(i, "./figDesc/div/p/s", namespace) else []}
            try:
                table_dict["coords"] = ([j.attrib.get("coords") for j in i.findall("./figDesc/div/p/s", namespace)])
                table_dict["table_coords"] = ([j.attrib.get("coords") for j in i.findall("./table", namespace)])
                tables = i.findall("./table", namespace)
                # 获取table每个row
                tables_rows = []
                # \multicolumn{3}{c|}{bbbb}
                for table in tables:
                    table_row = []
                    rows = table.findall("./row", namespace)
                    for row in rows:
                        row_items = []
                        cells = row.findall("./cell", namespace)
                        for cell in cells:
                            cols = int(cell.attrib.get("cols") if "cols" in cell.attrib else "1")
                            tmp_text = convert_latex_text(cell.text if cell.text else "")
                            if cols > 1:
                                tmp_text = f"\\multicolumn{{{cols}}}{{|l|}}{{{tmp_text}}}"
                            row_items.append(tmp_text)
                        table_row.append(" & ".join(row_items) + " \\\\ \\hline")
                    tables_rows.append(table_row)
                table_dict["tables_rows"] = tables_rows
            except Exception as e:
                logging.exception(e)
            table_list.append(table_dict)

        if i.attrib.get("type") == "figure" or "fig" in i.attrib.get(namespace_key):
            figure_dict = {"head": try_get_text_from_xml(i, "./head", namespace)
            if try_get_text_from_xml(i, "./head", namespace) else "",
                           "label": try_get_text_from_xml(i, "./label", namespace)
                           if try_get_text_from_xml(i, "./label", namespace) else "",
                           "figure_title": try_get_all_text_from_xml(i, "./figDesc/div/p/s", namespace)
                           if try_get_all_text_from_xml(i, "./figDesc/div/p/s", namespace) else []}
            figure_list.append(figure_dict)
    return table_list, figure_list


def concat_figure(mmd_text, figure):
    if len(figure["figure_title"]) > 0:
        title = "".join(figure["figure_title"])
    else:
        title = ""
    mmd_text += "\\begin{figure}\n"
    mmd_text += "\\caption{" + title + "\n"
    mmd_text += "\\label{" + figure.get("label", "") + "}\n"
    mmd_text += "\\head{" + figure.get("head", "") + "}}\n"
    mmd_text += "\\end{figure}\n"
    return mmd_text


def concat_table(mmd_text, table):
    if len(table["table_title"]) > 0:
        title = "".join(table["table_title"])
    else:
        title = ""
    mmd_text += "\\begin{table*}[h]\n"
    mmd_text += "\\caption{" + title + "\n"
    if len(table.get("coords", [])) > 0:
        mmd_text += "\n".join(table.get("coords", [])) + "\n"
    mmd_text += "\\label{" + table.get("label", "") + "}\n"
    mmd_text += "\\head{" + table.get("head", "") + "}}\n"
    table_rows = table.get("tables_rows", [])
    if len(table_rows) > 0:
        mmd_text += "\\begin{tabular*}\n"
        mmd_text += "\n".join(table_rows) + "\n"
        mmd_text += "\\end{tabular*}\n"
    mmd_text += "\\end{table*}\n"
    return mmd_text


def write_pdf(encoded_string, pdf_id):
    path = str(pdf_id) + ".pdf"
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(base64.b64decode(encoded_string))


def write_mmd(mmd_text, result_path):
    with open(result_path, 'w+') as f:
        f.write(mmd_text)


def read_file(file_path):
    with open(file_path) as file_obj:
        contents = file_obj.read()
    return contents


def request_grobid(pdf_id):
    # pdf_name = str(pdf_id) + ".pdf"
    # client = GrobidClient(config_path="./config.json")
    # res = client.process("processFulltextDocument", ".", output="./", tei_coordinates=True, force=True)
    # print(res)
    res = os.system(
        "grobid_client --input ./" + str(pdf_id) + "/ --output ./"
        + str(pdf_id) + "/ --config grobid/config.json --teiCoordinates --segmentSentences processFulltextDocument")
    if res != 0:
        raise Exception("Request to grobid error.")


def parse_table(xml_path):
    table_list = []
    # 读取文件
    dom = parse(xml_path)
    # 获取文档元素对象
    data = dom.documentElement
    # 获取 student
    tables = data.getElementsByTagName('figure')
    for table in tables:
        # 获取标签属性值
        figure_type = table.getAttribute('type')
        target = table.getAttribute('xml:id')
        # print(figure_type)
        if (figure_type == 'table'):
            table_latex = ''
            table_latex = table_latex + '\\begin{table*}[h]' + '\n' + '\caption{' + '\n' + '<p>' + '\n'
            # 表格标题等信息
            head = ''
            label = ''
            if len(table.getElementsByTagName('head')[0].childNodes) != 0:
                head += table.getElementsByTagName('head')[0].childNodes[0].nodeValue
            if len(table.getElementsByTagName('label')[0].childNodes) != 0:
                label += table.getElementsByTagName('label')[0].childNodes[0].nodeValue
            if (len(table.getElementsByTagName('figDesc')[0].getElementsByTagName('div')) != 0 and
                    len(table.getElementsByTagName('figDesc')[0].getElementsByTagName('div')[0].getElementsByTagName(
                        'p')) != 0):
                figDescs = \
                    table.getElementsByTagName('figDesc')[0].getElementsByTagName('div')[0].getElementsByTagName('p')[
                        0].getElementsByTagName('s')
                for figDesc in figDescs:
                    if not figDesc.childNodes[0].nodeValue:
                        value = ""
                    else:
                        value = figDesc.childNodes[0].nodeValue
                    table_latex = table_latex + '<s>' + '\n' + 'coords="' + figDesc.getAttribute(
                        'coords') + '"' + '\n' + value + '\n' + '<\s>' + '\n'
            table_latex = table_latex + '<\p>' + '\n'
            table_latex += '<head>' + head + '<\\head>' + '\n'
            table_latex += '<label>' + label + '<\\label>' + '\n'
            table_latex += '<target>' + target + '<\\target>' + '\n'
            table_latex += '\\label{tbl:' + label + "}}" + '\n'

            rows = []

            rows_xml = table.getElementsByTagName('row')
            for row_xml in rows_xml:
                row = []
                cells = row_xml.getElementsByTagName('cell')
                for cell in cells:
                    try:
                        cell_value = cell.childNodes[0].nodeValue
                    except Exception:
                        cell_value = 'empty'
                        # print(cell.childNodes[0].nodeValue)
                    try:
                        cols = int(cell.getAttribute('cols'))
                    except Exception:
                        cols = 1
                    if cols > 1:
                        for i in range(cols):
                            row.append(cell_value)
                    else:
                        row.append(cell_value)

                rows.append(row)
            # table = ''
            # for row in rows:
            #     table += "| " + " | ".join(row) + " |\n"
            # print(table)
            # print(rows)
            if (len(rows) > 1 and len(rows[0]) > 0):
                columns_len = len(rows[0])
                tmp_rows = []
                for row in rows[1:]:
                    if row is None:
                        continue
                    if len(row) > columns_len:
                        tmp_rows.append(row[0: columns_len])
                    elif len(row) < columns_len:
                        tmp_rows.append(row + [" "] * (columns_len - len(row)))
                    else:
                        tmp_rows.append(row)
                df = pd.DataFrame(tmp_rows, columns=rows[0])
                table_latex = table_latex + df.to_latex(index=False)
            table_latex = table_latex + '\\end{table*}' + '\n'
            table_list.append(table_latex)
    return table_list


def parse_figure(xml_path):
    figure_list = []
    # 读取文件
    dom = parse(xml_path)
    # 获取文档元素对象
    data = dom.documentElement
    # 获取 student
    figures = data.getElementsByTagName('figure')
    for figure in figures:
        # 获取标签属性值
        figure_type = figure.getAttribute('type')
        target = figure.getAttribute('xml:id')
        if (figure_type != ''):
            continue
        figure_latex = ''
        figure_latex = figure_latex + '\\begin{figure}' + '\n' + '\caption{' + '\n' + '<p>' + '\n'
        # 表格标题等信息
        head = ''
        label = ''
        if len(figure.getElementsByTagName('head')[0].childNodes) != 0:
            head += figure.getElementsByTagName('head')[0].childNodes[0].nodeValue
        if len(figure.getElementsByTagName('label')[0].childNodes) != 0:
            label += figure.getElementsByTagName('label')[0].childNodes[0].nodeValue
        if (len(figure.getElementsByTagName('figDesc')[0].getElementsByTagName('div')) != 0 and
                len(figure.getElementsByTagName('figDesc')[0].getElementsByTagName('div')[0].getElementsByTagName(
                    'p')) != 0):
            figDescs = \
                figure.getElementsByTagName('figDesc')[0].getElementsByTagName('div')[0].getElementsByTagName('p')[
                    0].getElementsByTagName('s')
            for figDesc in figDescs:
                if not figDesc.childNodes[0].nodeValue:
                    value = ""
                else:
                    value = figDesc.childNodes[0].nodeValue
                figure_latex = figure_latex + '<s>' + '\n' + 'coords="' + figDesc.getAttribute('coords') + '"' + '\n' + \
                               value + '\n' + '<\s>' + '\n'

        figure_latex = figure_latex + '<\p>' + '\n'
        figure_latex += '<head>' + head + '<\\head>' + '\n'
        figure_latex += '<label>' + label + '<\\label>' + '\n'
        figure_latex += '<target>' + target + '<\\target>' + '\n'
        figure_latex += '\\label{fig:' + label + "}}" + '\n'
        figure_latex += '\\end{figure}\n'
        figure_list.append(figure_latex)

    return figure_list


def main(pdf_path, data, random_uuid):
    paddleocr_dict = ""
    try:
        logging.info(data["path"] + " process start.")
        paddleocr_dict = paddleocr_process.paddle_process(pdf_path, data, random_uuid)
        result_path = pdf_path.replace(str(data["id"]) + ".pdf", str(data["id"]) + ".mmd")
        id = str(data["id"])
        if not os.path.exists(pdf_path):
            raise Exception(pdf_path + " not found.")
        logging.info("run grobid......")
        if check_pdf_scanned(pdf_path):
            raise Exception(pdf_path + " is scanned pdf.")
        request_grobid(random_uuid)
        logging.info(pdf_path + " request grobid end.")

        mmd_text = ""
        tree = ET.parse(random_uuid + "/" + id + ".grobid.tei.xml")
        xml_content = read_file(random_uuid + "/" + id + ".grobid.tei.xml")

        # 处理标题与元数据
        logging.info("get_meta......")
        meta = get_meta(namespace, tree)
        mmd_text += "#Title\n" + str(meta.get("Title", "")) + "\n"
        mmd_text += "#Meta\n"
        mmd_text += "##Language\n" + str(meta.get("Language", "")) + "\n"
        mmd_text += "##Keywords\n" + str(meta.get("Keywords", "")) + "\n"

        # 处理目录
        logging.info("content_process......")
        content_text = content_process(xml_content)

        # 处理图
        figure_text = "##Figures\n"
        logging.info("parse_figure......")
        figure_list = parse_figure(random_uuid + "/" + id + ".grobid.tei.xml")
        for figure in figure_list:
            figure_text += figure

        # 处理表
        logging.info("parse_table......")
        table_text = "##Tables\n"
        # for i in table_list:
        #    table_text = concat_table(table_text, i)
        table_list = parse_table(random_uuid + "/" + id + ".grobid.tei.xml")
        for table in table_list:
            table_text += table

        # 处理引用
        logging.info("get_reference......")
        refs_text = "##Reference\n" + get_references(random_uuid + "/" + id + ".grobid.tei.xml") + "\n"

        # 处理正文
        logging.info("body_process......")
        main_body_text = body_process(xml_content, "paper")

        mmd_text += content_text + figure_text + table_text + refs_text + main_body_text
        logging.info("inline_math begin...")
        mmd_text_inline_math = inline_math(mmd_text, random_uuid + "/" + id + ".pdf")
        logging.info("mmd_text_inline_math")

        logging.info("xml_content_text begin......")
        xml_content_text = revise_coodinates_str(xml_content, mmd_text_inline_math)
        logging.info("xml_content_text")

        # 后处理
        after_mmd = after_process.main(xml_content_text)

        # 写入本地文件
        write_mmd(after_mmd, result_path)
        logging.info(pdf_path + " get mmd end.")
        # upload_file("text_extraction_paper_files/" + str(version) + "/" + id + ".mmd",
        #             random_uuid + "/" + id + ".mmd")
        # upload_file("text_extraction_paper_files/" + str(version) + "/" + id + ".grobid.tei.xml",
        #             random_uuid + "/" + id + ".grobid.tei.xml")
        upload_xml_url = ("basic/arxiv/" + data["version"] + "/" + str(random_uuid)
                           + "/" + str(random_uuid) + ".grobid.tei.xml")
        utils.upload_file(upload_xml_url, random_uuid + "/" + id + ".grobid.tei.xml")
        logging.info(pdf_path + " process success.")
        model = fasttext.load_model("/workspace/subject_textclf/classifier_multi_subject.bin")
        subject = utils.get_subject(after_mmd, model)
        paddleocr_dict['subject'] = subject
        return str(paddleocr_dict), True, random_uuid + "/" + id + ".mmd", "success"

    except Exception as e:
        logging.error(pdf_path + " process failed. Error: " + str(e))
        return str(paddleocr_dict), False, "", str(e)


if __name__ == "__main__":
    print(main(
        "scihub_unzip/00100000/libgen.scimag00100000-00100999/10.1002/%28sici%291099-1573%28200005%2914%3A3%3C163%3A%3Aaid-ptr588%3E3.0.co%3B2-d.pdf",
        "jm970034q", "1.0"))
    # upload_file("text_extraction_files/123.mmd", "123.mmd")
    # del_temp_files(["123.mmd", "123.json"])
