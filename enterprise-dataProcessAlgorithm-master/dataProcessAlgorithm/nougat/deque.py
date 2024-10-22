import re


def do_dedup(text):
    lines = text.split("\n")
    is_table = False
    is_formula = False
    is_figure = False
    threshold = 4
    out_lines = []
    all_dup = []
    missing_pattern = r"\[MISSING_PAGE_\w+:\d+\]"
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
            if wd == "[START_TABLE]":
                is_table = True
                i += 1
                continue
            if wd == "[END_TABLE]":
                is_table = False
                i += 1
                continue
            elif wd == "[START_FORMULA]":
                is_formula = True
                i += 1
                continue
            elif wd == "[END_FORMULA]":
                is_formula = False
                i += 1
                continue
            elif wd == "[START_FIGURE]":
                is_figure = True
                i += 1
                continue
            elif wd == "[END_FIGURE]":
                is_figure = False
                i += 1
                continue
            elif re.search(missing_pattern, wd):
                i += 1
                continue
            if is_table or is_formula or is_figure:
                i += 1
                continue

            pos = i
            while pos < len(words):
                if words[pos] == wd:
                    pos += 1
                else:
                    break
            dup_num = pos - i
            if dup_num > threshold:
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
            if dup_num > threshold:
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
            if dup_num > threshold:
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
            if dup_num > threshold:
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


def split_tag(word):
    new_items = []
    items = word.split("[")
    if len(items) <= 1:
        return [word]
    new_items.append(items[0])
    for i in range(1, len(items)):
        item = items[i]
        new_items.append("[" + item)
    new_items2 = []
    for item in new_items:
        if item == "":
            continue
        sacs = item.split("]")
        if len(sacs) <= 1:
            new_items2.append(item)
            continue
        for j in range(len(sacs)):
            if j == len(sacs) - 1:
                if sacs[j] != "":
                    new_items2.append(sacs[j])
            else:
                new_items2.append(sacs[j] + "]")
    return new_items2


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


if __name__ == "__main__":
    text = "dsafdsa dsfdsafdf dsfdsaf [START_FORMULA]dsfdsafdsafdsafdsafd dsafdsafds dsfdsdsa[END_FORMULA] " \
           "1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 " \
           "1994 1994 1994 1994 1994 1994 1994 1994 1994 1994      1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 1994 " \
           "1994 1994 1994  1994 1994 1994 1994 1994 1994 1994 1994 dsafdsd dsafdsa dsafdsdsa dsfds\n" \
           "augest 14 augest 14 augest 14 augest 14 augest 14 augest 14 augest 14 augest 14 augest 14 augest 14 " \
           "augest 14 augest 14 augest 14 ehew\n" \
           "hello world papa hello world papa hello    world papa   hello world papa        hello world papa hello world papa hello world papa hello world papa hello world papa hello world papa [START_FORMULA]\n" \
           "nishishui lala wow me zhang lala wow me zhang[END_FORMULA] lala wow me zhang lala  wow me  zhang lala wow me zhang lala wow me  zhang lala wow me    zhang lala wow me  zhang nihao\n" \
           "dafdsf \\hello? \\hello? \\hello?  \\hello?  \\hello?   \\hello? \\hello? \\hello?  \\hello? \\hello? \\hello? dsfdsafsd dsafds dsafdas dsafds\n" \
           "[]"
    print(text)
    text, dup_str = do_dedup(text)
    print()
    print("text", text)
    print()
    for d in dup_str:
        print("dup_str", d)
