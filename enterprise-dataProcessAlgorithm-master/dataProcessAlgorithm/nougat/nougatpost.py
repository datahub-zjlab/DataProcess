import logging
import re
from typing import Tuple

import langid
import fasttext
import utils
from nougat.deque import do_dedup
from nougat.nonsense_cleaning import NonsenseCleaning


def title_pro(text):
    if len(text) > 0:
        lines = text.strip().split('\n')
        for i in range(min(len(lines), 3)):
            if re.search(r'^MISSING_PAGE', lines[i]) is None:
                if re.search(r'^([#*]+)\s*\w+', lines[i]):
                    if len(lines[i].strip('#').strip('*').split()) > 3:
                        return lines[i].strip('#').strip('*').strip()
                # elif len(lines[i].strip('#').strip('*').split()) > 3 and re.search(r'\({}\^{\star}\)', lines[i]) is None:
                #     return lines[i].strip('#').strip('*')
    return ''


def language_pro(text):
    if len(text) > 0:
        return langid.classify(text)[0]
    return ''


def keyword_pro(text):
    if len(text) == 0:
        return ''
    lines = text.split('\n')
    for i in range(len(lines)):
        if re.search(r'Keywords', lines[i]):
            tem_line = re.sub(r'Keywords', '', lines[i])
            keyword_list = tem_line.strip('#').strip('*').strip(',').strip(':').split(',')
            keyword_list = ['\'' + el + '\'' for el in keyword_list]
            return '[' + ','.join(keyword_list) + ']'
    return ''


def content_pro(text):
    if len(text) == 0:
        return ''
    lists = text.strip().split('\n')
    result = []
    for line in lists:
        if re.search(r"^#+\s*Abstract", line):
            result = []
            continue
        if re.search(r'^#{2,}\s*', line):
            match = re.search(r'^(#{2,})\s*', line)
            level = len(match.group(1).strip()) - 2
            content = line[match.span()[1]:].strip()
            item = '\t' * level + '* ' + content
            result.append(item)

    return '\n'.join(result)


def figure_pro(text):
    if len(text) == 0:
        return ''
    result = []
    lines = text.strip().split('\n')
    for line in lines:
        if re.search(r'^Figure', line):
            match = re.search(r'Figure\s+([A.\d]+)\s*[:.]\s*', line, re.I)
            if match:
                cap_str = line[match.span()[1]:].strip().strip('*')
                result.append('\\begin{figure}\n\\caption{\n' + cap_str + '\n}\n\\label{fig:' + str(
                    match.group(1)) + '}\n\\end{figure}')
            else:
                cap_match = re.search(r'Figure\s+[:.]\s*', line, re.I)
                if cap_match:
                    cap_str = line[match.span()[1]:].strip().strip('*')
                    result.append(
                        '\\begin{figure}\n\\caption{\n' + cap_str + '\n}\n\\label{fig:}\n\\end{figure}')
    return '\n'.join(result)


def table_pro(text):
    if len(text) == 0:
        return ''
    result = ''
    lines = text.strip().split('\n')
    for i in range(len(lines)):
        if re.search(r'^\\begin{table}', lines[i]):
            tem_str = ''
            i += 1
            while not re.search(r'^\\end{table}', lines[i]):
                tem_str += lines[i] + '\n'
                i += 1
                if i >= len(lines):
                    break
            i += 1
            if i < len(lines):
                match = re.search(r'Table\s*(\d+)\s*:\s*(.+)', lines[i])
                if match:
                    result += '\\begin{table*}[h]\n\\caption{\n' + match.group(2) + '\n}\n\\label{tbl:' + match.group(
                        1) + '}\n'
                    i += 1
                else:
                    result += '\\begin{table*}[h]\n\\caption{\n}\n\\label{tbl:}\n'
            else:
                result += '\\begin{table*}[h]\n\\caption{\n}\n\\label{tbl:}\n'
            if len(tem_str.strip()) > 0:
                result += tem_str + '\\end{table*}\n'
            else:
                result += '\\end{table*}\n'
    return result


def digit_check(line):
    # 计数数字字符
    digit_count = sum(c.isdigit() for c in line)

    # 计算比例
    ratio = digit_count / len(line) if len(line) > 0 else 0

    return ratio


def ref_pro(text):
    if len(text) == 0:
        return ''
    lines = text.split("\n")
    result = []
    count = 0
    tmp_str = []
    ref_flag = False
    for line in lines:
        if len(line) == 0:
            continue
        if re.search(r'#+ References', line):
            ref_flag = True
            continue
        if ref_flag and re.search(r'^\**\s*\[\d+]\s*\w+', line):
            result.append(line)
            continue
        if ref_flag and re.search(r'\**\[*\s*(\w[\w\s])*\s*(et al.)*\s*\((\d{4}\w*)\)\]*', line):
            result.append(line)
            continue

        # if ref_flag and re.search(r'^*', line):
        #     result.append(line)
        #     continue

        if not ref_flag and re.search(r'^\**\s*\[\d+]\s*\w+', line):
            count += 2
            tmp_str.append(line)
            ref_flag = True

        if not ref_flag and re.search(r'\**\[*\s*(\w[\w\s])*\s*(et al.)*\s*\((\d{4}\w*)\)\]*', line):
            count += 2
            tmp_str.append(line)
            ref_flag = True

        if count > 4:
            result += tmp_str
            tmp_str = []
            count = 0
        # count-=1
        # if count<0:
        #     count=0
        #     ref_flag=False

    for i in range(len(result)):
        result[i] = result[i].strip('*_').strip()
        match = re.search(r"\[(\d+)\]\s*", result[i])
        if match:
            result[i] = f'{i}.[' + match.group(1) + '](' + result[i][match.span()[1]:] + ')'
        else:
            result[i] = f'{i}.(' + result[i] + ')'
    return '\n'.join(result)


def process_abstract(text):
    result = []
    data_list = text.split("\n")
    skip_flag = False
    for data_line in data_list:
        title = title_pro(text)
        if len(title) > 0 and title in data_line:
            continue
        if re.search(r"^#+\s*Abstract", data_line):
            result = [data_line]
        elif re.search(r"^Table\s*\d+\s*:", data_line):
            continue
        elif re.search(r"^\\begin{table}", data_line):
            skip_flag = True
            continue
        elif re.search(r"^\\end{table}", data_line):
            skip_flag = False
            continue
        elif re.search(r'^Figure\s+[A.\d]+\s*[:.]\s*', data_line):
            continue
        elif re.search(r"^#+ References", data_line):
            continue
        elif re.search(r"^\**\s*\[\d+]\s*\.+", data_line):
            continue
        elif re.search(r"^\**Keywords\**\s*:\s*", data_line, re.I):
            continue
        elif re.search(r'^\**\s*\[\d+]\s*\w+', data_line):
            continue
        elif re.search(r'\**\[*\s*(\w[\w\s])*\s*(et al.)*\s*\((\d{4}\w*)\)\]*', data_line):
            continue
        elif re.search(r'^Footnote\s*', data_line):
            continue
        elif re.search(r'Publisher', data_line) and digit_check(data_line) > 0.2:
            continue
        # elif re.search(r"\[MISSING_PAGE", data_line):
        #     continue
        elif skip_flag:
            continue
        else:
            result.append(data_line)
    return "\n".join(result)


def mainbody_pro(text):
    if len(text) == 0:
        return ''
    # try:
    #     # text = HeadCleaning.head_cleaning(text)
    #     text = CertainParagraphCleaning.certain_paragraph_cleaning(text)
    # except Exception as e:
    #     print(f"certain_paragraph_cleaning_exception: {e}")
    #     return ''
    try:
        text = process_abstract(text)
    except Exception as e:
        logging.error(f"process_abstract_exception: {e}")

    lines = text.split("\n")
    result = []
    abstract_flag = False
    for i in range(len(lines)):
        if len(lines[i]) == 0:
            continue

        if re.search(r'^#+\s*Abstract\s*', lines[i]):
            abstract_flag = True
            continue
        if abstract_flag and not re.search(r'^[*#]+', lines[i]):
            result.append('[START_ABSTRACT]\n' + lines[i] + '\n[END_ABSTRACT]')
            abstract_flag = False
            continue
        else:
            result.append(lines[i])
    return '\n'.join(result)


def meta_process(orig_text):
    if len(orig_text) == 0:
        return False, '', 'nougat_post文件文本内容为空！'
    text = ''
    message = ''
    isSuccess = True
    try:
        text, all_dup = do_dedup(orig_text)
    except Exception as e:
        isSuccess = False
        message += f"nougat post do_dedup_exception: {e}"
        logging.exception(e)
        return isSuccess, '', message

    try:
        text = NonsenseCleaning.nonsense_cleaning(text)
    except Exception as e:
        isSuccess = False
        message += f"nougat post nonsense_cleaning_exception: {e}"
        logging.exception(e)
        return isSuccess, '', message

    # meta
    meta_info = '\n'
    try:
        meta_info += title_pro(text)
        meta_info += '\n=============\n\n#Meta\n##Language\n'
        meta_info += language_pro(text)
        meta_info += '\n##Keywords\n'
        meta_info += keyword_pro(text)
        meta_info += '\n##Contents\n'
        meta_info += content_pro(text)
        meta_info += '\n##Figures\n'
        meta_info += figure_pro(text)
        meta_info += '\n##Tables\n'
        meta_info += table_pro(text)
        meta_info += '\n##Reference\n'
        meta_info += ref_pro(text)
        meta_info += '\n\n#Main Body\n'
        meta_info += mainbody_pro(text)
    except Exception as e:
        isSuccess = False
        message += f"nougat post pro exception: {e}"
        logging.exception(e)
    return isSuccess, meta_info, message


def nougat_post(input_path):
    if len(input_path) == 0:
        return False, '', 'nougat_post输入文件路径为空！'
    message = ''
    isSuccess = True
    subject = ''
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()
        isSuccess, text, message = meta_process(text)
        if not isSuccess:
            return False, message
        with open(input_path, 'w') as file:
            file.write(text)
        model = fasttext.load_model("/workspace/subject_textclf/classifier_multi_subject.bin")
        subject = utils.get_subject(text, model)

    except Exception as e:
        isSuccess &= False
        message += ('nougat_post Exception:' + str(e))

    return isSuccess, message, subject
