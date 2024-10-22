import re
import time

UPPERCASE_THRESH = 0.6
SYMBOL_DIGIT_THRESH = 0.6
URL_THRESH = 0.3
PARAGRAPH_LEN_THRESH = 20
CAPITALIZE_WORD_THRESH = 0.6

url_pattern = re.compile(r'https?://\S+|www\.\S+')  # (r'^www.([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*/?$')
email_pattern = re.compile(r"\w+@\w+\.\w+")
doi_pattern = re.compile(r"10\.[0-9]{4,}\/[-._;()\/:a-zA-Z0-9]+")


def do_dedup(text):
    lines = text.split("\n")
    threhold = 4
    out_lines = []
    all_dup = []
    for line in lines:
        tmp_words = line.split()
        words = []
        for wd in tmp_words:
            tags = re_split_tag(wd)
            words += tags
        # print("words", words)
        i = 0
        dup_set = set()
        dup_pattern_set = set()
        while i < len(words):
            wd = words[i]

            pos = i
            while pos < len(words):
                if words[pos] == wd:
                    pos += 1
                else:
                    break
            dup_num = pos - i
            if dup_num > threhold:
                dup_str = " ".join((dup_num - 1) * [wd])
                dup_set.add(dup_str)
                dup_pattern = "(" + reg_trans(wd) + "\\s+){" + str(dup_num - 1) + "}"
                dup_pattern_set.add(dup_pattern)
                i = pos
                continue

            pos = i
            if i + 1 >= len(words):
                i += 1
                continue
            bi_gram_str = " ".join(words[i:i + 2])
            while pos < len(words):
                if pos + 1 >= len(words):
                    break
                tmp_bi_gram = " ".join(words[pos:pos + 2])
                if tmp_bi_gram == bi_gram_str:
                    pos += 2
                else:
                    break
            dup_num = int((pos - i) / 2)
            if dup_num > threhold:
                dup_str = " ".join((dup_num - 1) * [bi_gram_str])
                dup_set.add(dup_str)

                dup_pattern = r"(" + reg_trans(words[i]) + "\\s+" + reg_trans(words[i + 1]) + "\\s+){" + str(
                    dup_num - 1) + "}"
                dup_pattern_set.add(dup_pattern)

                i = pos
                continue

            pos = i
            if i + 2 >= len(words):
                i += 1
                continue
            tri_gram_str = " ".join(words[i:i + 3])
            while pos < len(words):
                if pos + 2 >= len(words):
                    break
                tmp_tri_gram = " ".join(words[pos:pos + 3])
                if tmp_tri_gram == tri_gram_str:
                    pos += 3
                else:
                    break
            dup_num = int((pos - i) / 3)
            if dup_num > threhold:
                dup_str = " ".join((dup_num - 1) * [tri_gram_str])
                dup_set.add(dup_str)
                dup_pattern = r"(" + reg_trans(words[i]) + "\\s+" + reg_trans(words[i + 1]) + "\\s+" + reg_trans(
                    words[i + 2]) + "\\s+){" + str(
                    dup_num - 1) + "}"
                dup_pattern_set.add(dup_pattern)
                i = pos
                continue

            pos = i
            if i + 3 >= len(words):
                i += 1
                continue
            four_gram_str = " ".join(words[i:i + 4])
            while pos < len(words):
                if pos + 3 >= len(words):
                    break
                tmp_four_gram = " ".join(words[pos:pos + 4])
                if tmp_four_gram == four_gram_str:
                    pos += 4
                else:
                    break
            dup_num = int((pos - i) / 4)
            if dup_num > threhold:
                dup_str = " ".join((dup_num - 1) * [four_gram_str])
                dup_set.add(dup_str)
                dup_pattern = r"(" + reg_trans(words[i]) + "\\s+" + reg_trans(words[i + 1]) + "\\s+" + reg_trans(
                    words[i + 2]) + "\\s+" + reg_trans(words[i + 3]) + "\\s+){" + str(
                    dup_num - 1) + "}"
                dup_pattern_set.add(dup_pattern)
                i = pos
                continue

            i += 1
        # print(f"dup_set:{dup_set}")
        # print(f"dup_pattern_set:{dup_pattern_set}")
        # all_dup += dup_set
        dup_pos = []
        # for dup_str in dup_set:
        #     tmp_dup_pos = line.find(dup_str)
        #     if tmp_dup_pos != -1:
        #         dup_pos.append([tmp_dup_pos, tmp_dup_pos+len(dup_str)])
        for dup_pattern in dup_pattern_set:
            match = re.search(dup_pattern, line)
            if match:
                sp = match.span()
                dup_pos.append([sp[0], sp[1]])
                all_dup.append(match.group())

        dup_pos.sort(key=lambda y: y[0])
        dup_pos_1 = []
        for i in range(len(dup_pos)):
            tmp_pos = dup_pos[i]
            if len(dup_pos_1) > 0 and dup_pos_1[-1][1] > tmp_pos[0]:
                prev_pos = dup_pos_1[-1]
                tmp_left = min(prev_pos[0], tmp_pos[0])
                tmp_right = max(prev_pos[1], tmp_pos[1])
                dup_pos_1[-1] = [tmp_left, tmp_right]
            else:
                dup_pos_1.append(tmp_pos)

        dedup_str = ""
        if len(dup_pos_1) > 0:
            dedup_str += line[:dup_pos_1[0][0]]
        else:
            out_lines.append(line)
            continue
        for i in range(len(dup_pos_1)):
            tmp_start = dup_pos_1[i][1]
            if i + 1 < len(dup_pos_1):
                tmp_end = dup_pos_1[i + 1][0]
                dedup_str += line[tmp_start:tmp_end]
            else:
                dedup_str += line[tmp_start:]
        out_lines.append(dedup_str)
    return "\n".join(out_lines), all_dup


def re_split_tag(word):
    pattern = r"(\[(START|END)_\w+\])"
    match = re.finditer(pattern, word)
    pos = [i.span() for i in match]
    if len(pos) == 0:
        return [word]
    items = []
    if pos[0][0] > 0:
        items.append(word[0:pos[0][0]])
    for i in range(len(pos)):
        items.append(word[pos[i][0]: pos[i][1]])
        if i + 1 < len(pos) and pos[i][1] < pos[i + 1][0]:
            items.append(word[pos[i][1]: pos[i + 1][0]])
        if i + 1 >= len(pos) and pos[i][1] < len(word):
            items.append(word[pos[i][1]:])
    return items


def reg_trans(word):
    pattern = word.replace("\\", "\\\\")
    pattern = pattern.replace("^", "\\^").replace("$", "\\$")
    pattern = pattern.replace("*", "\\*").replace("+", "\\+").replace("?", "\\?")
    pattern = pattern.replace("[", "\\[").replace("]", "\\]").replace("|", "\\|")
    pattern = pattern.replace("{", "\\{").replace("}", "\\}").replace("(", "\\(").replace(")", "\\)")
    return pattern


def nonsense_cleaning(file_content: str):
    """
    短句等无意义段落删除
    :param file_content: 文档文本内容字符串，以换行符'\n' 分割段落
    :return:
    """
    # Step 1: 划分段落
    paragraphs = file_content.split('\n')
    del_or_resv = [1] * len(paragraphs)  # 1: reserve; 0: delete
    token_flag = [0] * len(paragraphs)  # 标记段落是否包含formula、ref、table、figure或其区块

    # Step2: 循环处理每一个段落
    for idx, line in enumerate(paragraphs):
        if len(line.strip()) == 0:  # 空行
            continue

        if url(line) == 0:  # URL 占比很高，删除该段落，无论是否为 title、是否段落长短、是否包含REF等token
            del_or_resv[idx] = 0  # 删除该段落
            continue

        if "[START_REF]" in line or "[END_REF]" in line or \
                "[START_FORMULA]" in line or "[END_FORMULA]" in line:  # 不处理包含引用和公式的段落
            token_flag[idx] = 1
            continue

        # 非以上这些情况，使用规则进行清理
        # 首字母大写占比较高，放宽对短段落判断的阈值
        if capitalize_word(line) == 0 and paragraph_short(line, threshold=3 * PARAGRAPH_LEN_THRESH) == 0:
            del_or_resv[idx] = 0  # 删除段落
            continue

        # 满足（多个单词, 数字与-）格式也进行清理
        if re.match("^[a-zA-Z\- , ]+, [\d\- ,]+$", line) and len(line) < 100:
            del_or_resv[idx] = 0  # 删除段落
            continue

        if paragraph_short(line) == 0 and (  # 段落很短，并且
                uppercase(line) == 0 or  # 大写字符占比很高，或
                # NonsenseCleaning.pure_digit(line) or                    # 数字占比很高，或
                symbol_digit(line) == 0  # 符号占比很高，或
                # NonsenseCleaning.url(line) == 0 or                              # url占比很高
                # NonsenseCleaning.capitalize_word(line) == 0                  # 首字母大写单词占比高
        ):
            # 不要这个段落
            del_or_resv[idx] = 0  # 删除段落
            continue


def url(paragraph):
    """
    输入段落，判断url和邮箱占比
    :param paragraph:
    :param threshold:
    :return: -1: 无法判断，0: url占比过高，1: 正常句子
    """
    paragraph = paragraph.strip()
    if len(paragraph) == 0:
        return -1

    # 空格长度
    space_count = paragraph.count(' ')
    space_count += paragraph.count('\t')

    url_match = url_pattern.findall(paragraph)
    url_len = len("".join(url_match))

    email_match = email_pattern.findall(paragraph)
    email_len = len("".join(email_match))

    doi_match = doi_pattern.findall(paragraph)
    doi_len = len("".join(doi_match))

    ratio = (url_len + email_len + doi_len) / (len(paragraph) - space_count)
    if ratio > URL_THRESH:
        return 0
    else:
        return 1


def capitalize_word(paragraph):
    """
    判断纯非字母单词、单词首字母大写的单词数占比。
    :param paragraph:
    :return: -1: 无法判断，0: 大写字符占比过高，1: 正常句子
    """
    paragraph = paragraph.strip()
    if len(paragraph) == 0:
        return -1

    word_list = re.findall(r'\b\w+\b', paragraph)
    if len(word_list) == 0:
        return 0

    cnt = sum([1 if ((word[
                          0].isupper() and word != "START_FORMULA" and word != "END_FORMULA" and word != "START_REF" and word != "END_REF") or
                     word[0].isdigit() or word[0] == '_') else 0 for word in word_list])

    if cnt / len(word_list) > CAPITALIZE_WORD_THRESH:
        return 0
    else:
        return 1


def paragraph_short(paragraph, threshold=PARAGRAPH_LEN_THRESH):
    """
    输入段落，判断是否过短
    :param paragraph:
    :return: 0: 过短，1: 正常段落
    """
    # 删除开头、结尾的空格、tab、换行符
    paragraph = paragraph.strip()
    # 删除开头的非字母字符,数字,标点符号
    paragraph = paragraph.lstrip('0123456789-,.:; ')
    # 计算一句话的单词数
    word_count = len(paragraph.split(' '))
    if word_count < threshold:
        return 0
    else:
        return 1


def uppercase(paragraph):
    """
    输入段落，判断大写字符占比是否过高
    :param paragraph:
    :return: -1: 无法判断，0: 大写字符占比过高，1: 正常句子
    """
    # 删除开头、结尾的空格、tab、换行符
    paragraph = paragraph.strip()

    if len(paragraph) == 0:
        return -1

    upper_count = 0
    letter_count = 0
    for c in paragraph:
        if c.isupper():
            upper_count += 1
        if c.isalpha():
            letter_count += 1

    if letter_count == 0:  # 没有字母
        return -1
    if upper_count / letter_count > UPPERCASE_THRESH:
        return 0
    else:
        return 1


def symbol_digit(paragraph):
    """
    输入段落，判断符号和数字占比是否过高
    :param paragraph: 较短的句子
    :return: 0: 符号占比过高，1: 正常句子
    """
    # 删除开头、结尾的空格、tab、换行符
    paragraph = paragraph.strip()
    # symbol 表示非字母的字符，包括数字和符号
    symbol_count = 0
    # 字符串长度
    length = len(paragraph)
    # 字母长度
    letter_count = sum([1 for char in paragraph if char.isalpha()])
    # 空格长度
    space_count = paragraph.count(' ')
    space_count += paragraph.count('\t')
    # 其他字符长度
    symbol_count = length - letter_count - space_count
    # 计算占比
    symbol_ratio = (symbol_count) / (length - space_count)
    if symbol_ratio > SYMBOL_DIGIT_THRESH:
        return 0
    else:
        return 1


def figure_format(mmd_string):
    string_list = mmd_string.split('\n')
    html_clean = r'<\\?[sp]>'
    res = []
    caption_flag = 0
    caption_latex = []
    for str in string_list:
        if str.startswith('\\begin{figure}'):
            caption_flag = 1
            caption_latex.append(str)
            continue
        elif str.startswith('\\end{figure}'):
            caption_latex.append(str)
            res.extend(caption_latex)
            caption_latex = []
            caption_flag = 0
            continue
        if caption_flag == 1:
            str = re.sub(html_clean, '', str)
            str = re.sub("<head>.*<\\\\head>", "", str)
            str = re.sub("<label>.*<\\\\label>", "", str)
            str = re.sub("<target>.*<\\\\target>", "", str)
            if str and not str.startswith('coords='): caption_latex.append(str)
        else:
            res.append(str)
    return "\n".join(res)


def table_format(mmd_string):
    string_list = mmd_string.split('\n')
    html_clean = r'<\\?[sp]>'
    res = []
    caption_flag = 0
    caption_latex = []
    for str in string_list:
        if str.startswith('\\begin{table'):
            caption_flag = 1
            caption_latex.append(str)
            continue
        elif str.startswith('\\end{table'):
            caption_latex.append(str)
            res.extend(caption_latex)
            caption_latex = []
            caption_flag = 0
            continue
        if caption_flag == 1:
            str = re.sub(html_clean, '', str)
            str = re.sub("<head>.*<\\\\head>", "", str)
            str = re.sub("<label>.*<\\\\label>", "", str)
            str = re.sub("<target>.*<\\\\target>", "", str)
            if str and not str.startswith('coords='): caption_latex.append(str)
        else:
            res.append(str)
    return "\n".join(res)


def table_delete(mmd_string):
    string_list = mmd_string.split('\n')
    res = []
    table_flag = False
    for i in string_list:
        if i == "##Tables":
            table_flag = True
            res.append(i)
        elif i == "##Reference" or i == "#Main Body":
            table_flag = False
            res.append(i)
        elif table_flag:
            continue
        else:
            res.append(i)
    return "\n".join(res)


def revise_citation(mmd_text):
    citation_pattern = re.compile(r'(\d+\.\[\d+\].*?\))\n', re.S)  ####以1.[1]开头，且以)\n结尾的句子
    citation_results = citation_pattern.findall(mmd_text)
    for idx, cit in enumerate(citation_results):
        newcit = re.sub(r'\d+\.(\[.*?\()', '', cit)  ###去除多余id
        newcit = newcit[::-1].replace(')', '')[::-1]  ####去除最后一个小括号
        newcit.lstrip('.')
        newcit = '[' + str(idx + 1) + ']' + newcit  ####增加中括号
        mmd_text = mmd_text.replace(cit, newcit)  ####全文范围更换
    return mmd_text


def delete_figure(mmd_text):
    result_list = []
    if len(mmd_text.split("#Main Body")) >= 2:
        main_body = mmd_text.split("#Main Body")[-1]
        main_body_list = main_body.split("\n")
        tmp = []
        flag = False
        is_figure = False
        for text in main_body_list:
            if text == "<p>":
                flag = True
                tmp.append(text)
            elif text == "</p>":
                tmp.append(text)
                if not is_figure:
                    result_list.append("\n".join(tmp))
                tmp = []
                flag = False
                is_figure = False
            elif flag == True and re.search("Figure \d", text) and "ref type=\"figure\"" not in text:
                is_figure = True
                tmp.append(text)
            else:
                flag = False
                tmp.append(text)
    if len(result_list):
        return mmd_text.split("#Main Body")[0] + "#Main Body" + "\n".join(result_list)
    else:
        return mmd_text


def delete_tag_and_splice_paragraphs(mmd_text):
    result_list = []
    result_list_clean = []
    if len(mmd_text.split("#Main Body")) >= 2:
        main_body = mmd_text.split("#Main Body")[1]
        main_body_list = main_body.split("\n")

        tmp = ""
        for text in main_body_list:
            if '[START_ABSTRACT]' in text or '[END_ABSTRACT]' in text:
                result_list.append(text)
            elif text == "<p>" or text == "</p>":
                if len(tmp) > 0:
                    result_list.append(tmp)
                    tmp = ""
                continue
            elif re.match(r'#+\w', text):
                result_list.append('\n' + text)
                tmp = ""
            else:
                if len(tmp) == 0:
                    tmp += text.replace("<s>", "").replace("</s>", "")
                else:
                    tmp += " " + text.replace("<s>", "").replace("</s>", "")
        # 公式标签重复
        label_lists = re.findall(r'\\label{eq:\d+\.?\d*}', mmd_text)
        label_dict = {}
        for label in label_lists:
            label_num = label[10:-1]
            label_dict[label_num] = 1
        # 删除行头乱码字符
        for line in result_list:
            if re.match('•+', line):
                line = re.sub(r'•+', '', line)
            if re.match('\(\d+(\.?\d*)*\)$', line):
                if '[END_EQU]' in result_list_clean[-1]:
                    pos_insert = result_list_clean[-1].find('[END_EQU]')
                    result_list_clean[-1] = result_list_clean[-1][0:pos_insert] + line + '[END_EQU]'
                else:
                    result_list_clean[-1] += line
                text = result_list_clean[-1]
                pos_start = text.find('[START_EQU]')
                pos_end = text.find('[END_EQU]', pos_start)
                content = text[pos_start + 11:pos_end]
                if re.search('\(\d+(\.?\d*)*\)$', content):
                    formula_number = re.search('\(\d+(\.?\d*)*\)$', content).group()
                    content = content.replace(formula_number, '')
                    content = content.strip().strip(',')
                    formula_number = formula_number[1:-1]
                    if formula_number in label_dict:
                        new_formula_number = formula_number + label_dict[formula_number] * 'b'
                        label_dict[formula_number] += 1
                        formula_number = new_formula_number
                    label = 'eq:' + formula_number
                    pos_end = content.find('[START_EQU_FLAG]')
                    if pos_end != -1:
                        old_str = content.strip()[:pos_end]
                        new_str = old_str
                        if old_str[:2] == '\(':
                            new_str = new_str[2:]
                        if old_str[-2:] == '\)':
                            new_str = new_str[:-2]
                        content = content.replace(old_str, new_str)
                    formula_content = '\\begin{equation} ' + content + ' ' + '\\label{' + label + '}' + ' ' + '\\end{equation}\n'
                    pos_start = line.find('[START_EQU]')
                    pos_end = line.find('[END_EQU]', pos_start)
                    old_str = line[pos_start:pos_end]
                    # result_list_clean[-1] = result_list_clean[-1].replace(old_str,formula_content)
                    result_list_clean[-1] = formula_content
                continue
            result_list_clean.append(line)
    if len(result_list):
        return mmd_text.split("#Main Body")[0] + "#Main Body\n" + "\n".join(result_list_clean)
    else:
        return mmd_text


def main_body_convert(input_str: str) -> str:
    if len(input_str) == 0:
        return ""
    lines = input_str.strip().split('\n')
    results = []
    body_part = False
    pattern = r' coords="([^"]+)"'
    for line in lines:
        if re.match('#Main Body', line, re.I):
            body_part = True
            results.append(line)
            continue
        if body_part == False:
            results.append(line)
            continue
        if re.search(pattern, line):
            new_line = re.sub(pattern, '', line)
            results.append(new_line)
        elif re.match(r'#+\w', line):  # 标题前加换行
            title_lists = re.findall('#+\w', line)
            for title in title_lists:
                line = line.replace(title, '\n' + title)
            results.append(line)
        else:
            results.append(line)
    return '\n'.join(results)


def write_mmd(mmd_text, path, filename):
    filepath = path + filename
    with open(filepath, 'w+', encoding="utf-8") as f:
        f.write(mmd_text)


def read_mmd(mmd_path):
    with open(mmd_path, encoding="utf-8") as file_obj:
        contents = file_obj.read()
    return contents


def all_tag(sections, fig=False, tab=False, ref=False):
    if fig or tab == True:
        pattern = re.compile(r'<target>([\w_]+)<\\target>')
        label_pattern = re.compile(r'\\label\{(.*?)\}')
        idx = 1
    elif ref == True:
        pattern = re.compile(r'<b\d+>')
        label_pattern = re.compile(r'\[(.*?)\]')
        idx = 0
    result = {}
    for section in sections:
        label_match = label_pattern.search(section)
        match = pattern.search(section)
        if match and label_match:
            value = (match.group(idx))
            label = (label_match.group(idx))
            result[value] = label
    return result


########### read main part #############
def read_main_body(mmd_string):
    lines = mmd_string.split("\n")
    start_index = lines.index("#Main Body") + 1
    content_after_main_body = lines[start_index:]
    return content_after_main_body


########### read tables and split and fine the tag #############
def read_ref(mmd_string, start, end):
    lines = mmd_string.split("\n")
    start_index = lines.index(start) + 1
    end_index = lines.index(end)
    lines = lines[start_index:end_index]
    # sections = []
    # pattern = re.compile(r'^\d{1,4}')
    # for line in lines:
    #    if pattern.match(line):
    #        sections.append(line)
    #    else:pass
    return lines


########### read figures and split and fine the tag #############
def read_content(mmd_string, start, end):
    lines = mmd_string.split("\n")
    if start == "":
        end_index = lines.index(end)
        lines = lines[:end_index]
    else:
        start_index = lines.index(start) + 1
        end_index = lines.index(end)
        lines = lines[start_index:end_index]
    sections = []
    section = []
    for line in lines:
        if line.startswith('\\begin'):
            if section:
                sections.append('\n'.join(section))
            section = [line]
        else:
            section.append(line)
    if section:
        sections.append('\n'.join(section))
    return sections


def move_label_outof_caption(section):
    lines = section.split("\n")
    for i in reversed(range(len(lines))):
        if r"\label" in lines[i]:
            label = lines.pop(i)
            lines.insert(i + 1, label)
    return "\n".join(lines)


def link_revert(mmd_string):
    sections_useless = read_content(mmd_string, "", "##Figures")
    sections_fig = read_content(mmd_string, "##Figures", "##Tables")
    sections_tables = read_content(mmd_string, "##Tables", "##Reference")
    sections_ref = read_ref(mmd_string, "##Reference", "#Main Body")
    sections_main = read_main_body(mmd_string)

    sections_fig = adjust_section(sections_fig, "fig")
    sections_tables = adjust_section(sections_tables, "tab")
    targeted_fig = all_tag(sections_fig, fig=True)
    targeted_tab = all_tag(sections_tables, tab=True)
    targeted_ref = all_tag(sections_ref, ref=True)

    for i in range(len(sections_main)):
        temp_line = (sections_main[i])
        pattern = re.compile(r"<ref type=\"(.*?)\"")  # 什么类型的
        target_pattern = re.compile(r"target=\"(.*?)\"")  # 替换成什么
        replace_pattern = re.compile(r"<ref type=\".*?\">.*?</ref>")  # 整体部分
        replace_fig_pattern = re.compile(r"(Figure |Fig. |fig. | figure )?<ref type=\".*?\">.*?</ref>")
        replace_tbl_pattern = re.compile(r"(Table |tbl |table |tbl. )?<ref type=\".*?\">.*?</ref>")
        content_pattern = re.compile(r"<ref type=\".*?\">(.*?)</ref>")
        if "<ref type=" in temp_line:
            match = re.search(replace_pattern, temp_line)
            while match:
                pipeline = re.search(pattern, temp_line).group(1)
                temp_part = (match.group(0))
                if "target=" not in temp_part:
                    if pipeline == "bibr":
                        temp_line = re.sub(replace_pattern, "", temp_line, count=1)
                    else:
                        temp_line = re.sub(content_pattern, r"\1", temp_line, count=1)
                    match = re.search(replace_pattern, temp_line)

                else:
                    target = re.search(target_pattern, temp_line).group(1)[1:]
                    if pipeline == "bibr":
                        target = "<" + target + ">"
                        if target in targeted_ref.keys():
                            true_tag = "<START_REF>" + targeted_ref[target][1:-1] + "<END_REF>"
                            temp_line = re.sub(replace_pattern, true_tag, temp_line, count=1)
                        else:
                            temp_line = re.sub(replace_pattern, "", temp_line, count=1)
                    elif pipeline == "figure":
                        if target in targeted_fig.keys():
                            fig_content = re.findall(content_pattern, temp_line)
                            fig_former = f"[Figure {fig_content[-1]}]"
                            fig_latter = f"({targeted_fig[target]})"
                            true_targrt = "".join([fig_former, fig_latter]).replace("\\", "")
                            temp_line = re.sub(replace_fig_pattern, true_targrt, temp_line, count=1)
                        else:
                            temp_line = re.sub(content_pattern, r"\1", temp_line, count=1)
                    elif pipeline == "table":
                        if target in targeted_tab.keys():
                            tbl_content = re.findall(content_pattern, temp_line)
                            tbl_latter = f"({targeted_tab[target]})"
                            tbl_former = f"[Table {tbl_content[-1]}]"
                            tbl_target = "".join([tbl_former, tbl_latter])
                            # 转义错误
                            temp_line = temp_line.replace('\\', '/')
                            tbl_target = tbl_target.replace('\\', '/')
                            temp_line = re.sub(replace_tbl_pattern, tbl_target, temp_line, count=1)
                        else:
                            temp_line = re.sub(content_pattern, r"\1", temp_line, count=1)
                    else:
                        temp_line = re.sub(content_pattern, r"\1", temp_line, count=1)
                    match = re.search(replace_pattern, temp_line)

            sections_main[i] = temp_line

    sections_ref = ref_clear(sections_ref)

    for i in range(len(sections_fig)):
        sections_fig[i] = move_label_outof_caption(sections_fig[i])
    for i in range(len(sections_tables)):
        sections_tables[i] = move_label_outof_caption(sections_tables[i])

    res = ""
    for i in sections_useless:
        res += i
    res += "\n##Figures\n"
    for i in sections_fig:
        res += i
        res += "\n"
    res += "\n##Tables\n"
    for i in sections_tables:
        res += i
        res += "\n"
    res += "\n##Reference\n"
    for i in sections_ref:
        res += i
        res += "\n"
    res += "\n#Main Body\n"
    for i in sections_main:
        res += i
        res += "\n"
    res += "\n"
    return res


def ref_clear(sections):
    process_line = []
    tmp = ""
    for line in sections:
        pattern = r"<b\d+>[\n\t\s]*"
        line_clear = re.sub(pattern, "", line)
        line_clear = line_clear.strip("\t")
        if line.endswith(")"):
            process_line.append(tmp + line_clear)
            tmp = ""
        else:
            tmp += line_clear
    return process_line


def adjust_section(sections, key):
    pattern = re.compile(r'<target>([\w_]+)<\\target>')
    label_pattern = re.compile(r'\\label\{(.*?)\}')
    for i in range(len(sections)):
        section = sections[i]
        matchtest = pattern.search(section)  # 正确的
        label_matchtest = label_pattern.search(section)
        if matchtest and label_matchtest:
            match = pattern.search(section).group(1).split("_")[1]  # 正确的
            label_template = ''
            if key == "fig":
                label_template = "\\label{{fig:{}}}"
            elif key == "tab":
                label_template = "\\label{{tbl:{}}}"
            true_label = label_template.format(match)
            true_label = "\\" + true_label + "\n"
            sections[i] = label_pattern.sub(true_label, section)
    return sections


def main_body_formula_process(input_str: str) -> str:
    if len(input_str.split("#Main Body")) >= 2:
        main_body = input_str.split("#Main Body")[-1]
        main_body_list = main_body.split("\n")
        result_list = []
        formula_dict = {}
        formula_mark_num_dict = {}
        for text in main_body_list:
            if '<formula' in text:
                if 'target="' in text:
                    pos_start = text.find('>')
                    pos_end = text.find('</formula>', pos_start)
                    content = text[pos_start + 1:pos_end]
                    pos_start = text.find('target="')
                    pos_end = text.find('">', pos_start)
                    formula_key = text[pos_start + 8:pos_end]
                    label = 'eq:' + formula_key
                    formula_content = '\\begin{equation}\n' + content + '\n' + '\\label{' + label + '}' + '\n' + '\\end{equation}\n'
                    result_list.append(formula_content)
                    formula_dict[formula_key] = label
                else:
                    pos_start = text.find('>')
                    pos_end = text.find('</formula>', pos_start)
                    content = text[pos_start + 1:pos_end].strip()
                    # 有标
                    if re.search('\(\d+(\.?\d{0,4}){0,4}\)\[START_EQU_FLAG\]\<\d+\>\[END_EQU_FLAG\]$', content):
                        formula_content = re.search(
                            '\(\d+(\.?\d{0,4}){0,4}\)\[START_EQU_FLAG\]\<\d+\>\[END_EQU_FLAG\]$',
                            content).group()
                        pos_end = formula_content.find('[START_EQU_FLAG]<')
                        formula_number = formula_content[0:pos_end]

                        content = content.replace(formula_number, '')
                        content = content.strip().strip(',')
                        formula_number = formula_number[1:-1]
                        if formula_number in formula_mark_num_dict:
                            new_formula_number = formula_number + formula_mark_num_dict[formula_number] * 'a'
                            formula_mark_num_dict[formula_number] += 1
                            formula_number = new_formula_number
                        else:
                            formula_mark_num_dict[formula_number] = 1
                        label = 'eq:' + formula_number
                        formula_content = '\\begin{equation}\n' + content + '\n' + '\\label{' + label + '}' + '\n' + '\\end{equation}\n'
                        result_list.append(formula_content)
                    else:
                        replace_str = text[text.find('<formula'):text.find('>') + 1]
                        text = re.sub(replace_str, '[START_EQU]\(', text)
                        text = re.sub('</formula>', '[END_EQU]', text)
                        text = text.replace('[START_EQU_FLAG]', '\)[START_EQU_FLAG]')
                        result_list.append(text)
            else:
                result_list.append(text)
        # 正文ref type="formula" 替换
        result_list_ref = []
        for line in result_list:
            formula_ref_lists = re.finditer('<ref type="formula"', line)
            for formula_ref in formula_ref_lists:
                pos_start = formula_ref.start()
                pos_end = line.find('</ref>', pos_start)
                ref_str = line[pos_start:pos_end + 6]
                pos_start = ref_str.find('>')
                pos_end = ref_str.find('<', pos_start)
                link_name = ref_str[pos_start + 1:pos_end]
                pos_start = ref_str.find('target="#')
                pos_end = ref_str.find('">', pos_start)
                formula_key = ref_str[pos_start + 9:pos_end]
                if formula_key in formula_dict:
                    link_url = formula_dict[formula_key]
                    ref_replace = '[' + link_name + ']' + '(' + link_url + ')'
                    line = line.replace(ref_str, ref_replace)
            result_list_ref.append(line)
        return input_str.split("#Main Body")[0] + "#Main Body\n" + "\n".join(result_list_ref)
    else:
        return input_str


def change_title(text):
    result_list = []
    title_text = text.split("#Meta")[0]
    title_list = title_text.split("\n")
    for i in title_list:
        i = i.replace("#Title", "")
        if i == "None":
            i = ""
        result_list.append(i)
    return "\n".join(result_list) + "=============\n\n#Meta" + "\n".join(text.split("#Meta")[1:])


def process_inline_formula_line(line):
    if len(line) < 2:
        return line
    formula_index = []
    result = []
    find_formula = False
    num_brackets = 0
    for i in range(len(line) - 1):
        if (find_formula is False and line[i] == "\\" and
                (line[i + 1] == "(" or line[i + 1] == "[" or line[i + 1] == "{")):
            find_formula = True
            num_brackets += 1
            formula_index.append(i)
        elif (find_formula is True and line[i] == "\\" and
              (line[i + 1] == "(" or line[i + 1] == "[" or line[i + 1] == "{")):
            num_brackets += 1
        elif (find_formula is True and line[i] == "\\" and
              (line[i + 1] == ")" or line[i + 1] == "]" or line[i + 1] == "}")):
            num_brackets -= 1
            if num_brackets == 0:
                if i + 3 < len(line):
                    if (line[i + 2] == "\\" and
                            (line[i + 3] == "(" or line[i + 3] == "[" or line[i + 3] == "{")):
                        continue
                formula_index.append(i + 1)
                find_formula = False
                result.append(formula_index)
                formula_index = []
    line.find('[START_EQU_FLAG]<0>[END_END_FLAG]')
    new_line = ""
    if len(result) == 0:
        return line
    begin = -1
    last_index = [begin, begin]
    for index in result:
        pos_start = line.find('[START_EQU_FLAG]<', index[1])
        pos_end = line.find('>[END_EQU_FLAG]', pos_start)
        flag_str = line[pos_start:pos_end]
        new_line = new_line + line[begin + 1:index[0]] + "[START_EQU]" + line[
                                                                         index[0] + 2:index[
                                                                                          1] - 1] + flag_str + "[END_EQU]"
        begin = pos_end if pos_end != -1 else index[1]
        last_index = index
    new_line = new_line + line[last_index[1] + 1:]
    return new_line


def process_inline_formula(input_str):
    if len(input_str.split("#Main Body")) >= 2:
        main_body = input_str.split("#Main Body")[-1]
        main_body_list = main_body.split("\n")
        result = []
        for text in main_body_list:
            new_text = process_inline_formula_line(text)
            result.append(new_text)
        return input_str.split("#Main Body")[0] + "#Main Body\n" + "\n".join(result)
    else:
        return input_str


def process_inline_formula2(input_str):
    if len(input_str.split("#Main Body")) >= 2:
        main_body = input_str.split("#Main Body")[-1]
        inline_formulas = re.finditer(r'\\\(.*\\\)\[START_EQU_FLAG\]<\d+>\[END_EQU_FLAG\]', main_body)
        for f in inline_formulas:
            old_formula = f.group()
            new_formula = "[START_EQU]" + old_formula + "[END_EQU]"
            main_body = main_body.replace(old_formula, new_formula)
        inline_formulas = re.finditer(r'\\\[.*\\\]\[START_EQU_FLAG]<\d+>\[END_EQU_FLAG]', main_body)
        for f in inline_formulas:
            old_formula = f.group()
            new_formula = "[START_EQU]" + old_formula + "[END_EQU]"
            main_body = main_body.replace(old_formula, new_formula)
        return input_str.split("#Main Body")[0] + "#Main Body" + main_body
    else:
        return input_str


# 删除行间公式中的\()
def delete_line_formula_content_flag(input_str):
    if len(input_str.split("#Main Body")) >= 2:
        main_body = input_str.split("#Main Body")[-1]
        body_lines = main_body.split('\n')
        result = []
        for line in body_lines:
            if re.match('<formula .*>.*</formula>', line):
                pos_start = line.find('>')
                pos_end = line.find('</formula>')

                if pos_end != -1 and pos_start != -1:
                    old_str = line[pos_start + 1:pos_end]
                    new_str = old_str.strip()
                    line = line.replace(old_str, new_str)
                    pos_end = line.find('[START_EQU_FLAG]')
                    if pos_end != -1:
                        old_str = line[pos_start + 1:pos_end].strip()
                        new_str = old_str
                    if old_str[:2] == '\(':
                        new_str = new_str[2:]
                    if old_str[-2:] == '\)':
                        new_str = new_str[:-2]
                    line = line.replace(old_str, new_str)
            result.append(line)
        return input_str.split("#Main Body")[0] + "#Main Body" + "\n".join(result)
    else:
        return input_str


def main(mmd_string):
    # result = table_delete(mmd_string)  # 刘 元数据表删除
    result = delete_figure(mmd_string)  # 刘洋 删除正文中图的
    result = delete_line_formula_content_flag(result)  # 贾 删除行间公式中的\()
    result = process_inline_formula2(result)  # 贾 行内公式替换
    result = main_body_formula_process(result)  # 贾 公式处理
    result = link_revert(result)  # 劳 替换
    result = main_body_convert(result)  # 贾 正文处理
    result = figure_format(result)  # 舍 元数据图处理
    result = table_format(result)  # 刘 元数据表处理
    # result = revise_citation(result) #丁 元数据引用处理

    result = delete_tag_and_splice_paragraphs(result)  # 刘 删除<p>以及段落拼接 贾 拼接后公式处理
    result = change_title(result)

    return result
