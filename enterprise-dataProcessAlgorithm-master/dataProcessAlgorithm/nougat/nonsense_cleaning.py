
# encoding: utf-8
"""
@file: nonsense_cleaning.py
@time: 2023/10/24
@desc: 去除短句等无意义信息
   1. 句子中主要由大写字符组成（丢弃）
   2. 句子由纯数字组成（丢弃）
   3. 符号占比过高
   4. URL格式
   5. 段落长度过低
   2023年10月25日上述第一版代码跑出部分结果，讨论决定以下更新：
   1. 把 _ 开头和 _ 结尾的“标题”， _ 改为 **;
   2. 当URL的占比较高时，可忽略标题的判断，直接删除；
   3. 首字母大写单词的占比较高时，可放宽段落长度的判断，如占比较高且长度 < 3*PARAGRAPH_LEN_THRESH 时，可删除
   4. 删除的短段落之间的短段落，连续不超过3段，段落中不包含formula、ref、figure、table时，可删除
"""
import re
import nltk  # pip install nltk
from nltk.tokenize import sent_tokenize
import numpy as np


class NonsenseCleaning:
    UPPERCASE_THRESH = 0.6
    SYMBOL_DIGIT_THRESH = 0.6
    URL_THRESH = 0.3
    PARAGRAPH_LEN_THRESH = 5
    CAPITALIZE_WORD_THRESH = 0.6

    url_pattern = re.compile(r'https?://\S+|www\.\S+')  # (r'^www.([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*/?$')
    email_pattern = re.compile(r"\w+@\w+\.\w+")
    doi_pattern = re.compile(r"10\.[0-9]{4,}\/[-._;()\/:a-zA-Z0-9]+")

    @classmethod
    def segment_sentence(cls, file_content):
        """
        使用 nltk 库对文本内容进行句子分割，输出句子列表
        :param file_content:
        :return:
        """
        sentences = sent_tokenize(file_content)
        return sentences

    @classmethod
    def uppercase(cls, paragraph):
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
        if upper_count / letter_count > NonsenseCleaning.UPPERCASE_THRESH:
            return 0
        else:
            return 1

    @classmethod
    def capitalize_word(cls, paragraph):
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
        cnt = sum([1 if (word[0].isupper() or word[0].isdigit() or word[0] == '_') else 0 for word in word_list])

        if cnt / len(word_list) > cls.CAPITALIZE_WORD_THRESH:
            return 0
        else:
            return 1

    # @classmethod
    # def pure_digit(cls, paragraph, threshold=1.0):
    #     """
    #     输入段落，判断是否纯数字组成
    #     :param paragraph:
    #     :return: -1: 无法判断，0: 数字字符占比过高，1: 正常句子
    #     """
    #     pass

    @classmethod
    def symbol_digit(cls, paragraph):
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
        if symbol_ratio > NonsenseCleaning.SYMBOL_DIGIT_THRESH:
            return 0
        else:
            return 1

    @classmethod
    def url(cls, paragraph):
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

        url_match = cls.url_pattern.findall(paragraph)
        url_len = len("".join(url_match))

        email_match = cls.email_pattern.findall(paragraph)
        email_len = len("".join(email_match))

        doi_match = cls.doi_pattern.findall(paragraph)
        doi_len = len("".join(doi_match))

        ratio = (url_len + email_len + doi_len) / (len(paragraph) - space_count)
        if ratio > NonsenseCleaning.URL_THRESH and not re.search(r'^\**\s*\[\d+]\s*\w+', paragraph):
            return 0
        else:
            return 1

    @classmethod
    def paragraph_short(cls, paragraph, threshold=PARAGRAPH_LEN_THRESH):
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

    @classmethod
    def _is_title(cls, paragraph: str):
        """
        判断是否为文档标题，标题一般较短，不能作为短句删除
        :param paragraph:
        :return:
        """
        paragraph = paragraph.strip()
        # 标题一般由 # 或 * 开头
        if paragraph.startswith("#"):
            return True
        if paragraph.startswith("*") and paragraph.endswith("*"):
            return True
        if paragraph.startswith("_") and paragraph.endswith("_"):
            return True
        return False

    @classmethod
    def nonsense_cleaning(cls, file_content: str):
        """
        短句等无意义段落删除
        :param file_content: 文档文本内容字符串，以换行符'\n' 分割段落
        :return:
        """
        # Step 1: 划分段落
        paragraphs = file_content.split('\n')
        del_or_resv = [1] * len(paragraphs)  # 1: reserve; 0: delete
        hit_fig_tab_begin = False
        token_flag = [0] * len(paragraphs)  # 标记段落是否包含formula、ref、table、figure或其区块

        # Step2: 循环处理每一个段落
        for idx, line in enumerate(paragraphs):
            if hit_fig_tab_begin:  # 搜索 figure 或 table 的结束符
                if "[END_FIGURE]" in line or "[END_TABLE]" in line:
                    hit_fig_tab_begin = False
                token_flag[idx] = 1
                continue

            if len(line.strip()) == 0:  # 空行
                continue

            if "[MISSING_PAGE" in line:
                token_flag[idx] = 1
                continue

            if NonsenseCleaning.url(line) == 0:  # URL 占比很高，删除该段落，无论是否为 title、是否段落长短、是否包含REF等token
                del_or_resv[idx] = 0  # 删除该段落
                continue

            if NonsenseCleaning._is_title(line):  # 不处理标题
                if line.startswith("_") and line.endswith("_"):
                    line = line.strip("_")
                    line = "**" + line + "**"  # 将多个 _ 开头结尾的段落， _ 改为 **
                    paragraphs[idx] = line
                continue

            if "[START_REF]" in line or "[END_REF]" in line or \
                    "[START_FORMULA]" in line or "[END_FORMULA]" in line:  # 不处理包含引用和公式的段落
                token_flag[idx] = 1
                continue

            if "[START_FIGURE]" in line or "[START_TABLE]" in line:  # 遇到 figure 或 table
                hit_fig_tab_begin = True
                token_flag[idx] = 1
                continue
                
            if re.search(r'^\\begin{table}', line) or re.search(r'Figure\s+([A.\d]+)\s*[:.]\s*', line, re.I):  # 遇到 figure 或 table
                hit_fig_tab_begin = True
                token_flag[idx] = 1
                continue
            # 非以上这些情况，使用规则进行清理
            # 首字母大写占比较高，放宽对短段落判断的阈值
            if NonsenseCleaning.capitalize_word(line) == 0 and NonsenseCleaning.paragraph_short(line,
                                                                                                threshold=2 * cls.PARAGRAPH_LEN_THRESH) == 0:
                del_or_resv[idx] = 0  # 删除段落
                continue

            if NonsenseCleaning.paragraph_short(line) == 0 and (  # 段落很短，并且
                    NonsenseCleaning.uppercase(line) == 0 or  # 大写字符占比很高，或
                    # NonsenseCleaning.pure_digit(line) or                    # 数字占比很高，或
                    NonsenseCleaning.symbol_digit(line) == 0  # 符号占比很高，或
                    # NonsenseCleaning.url(line) == 0 or                              # url占比很高
                    # NonsenseCleaning.capitalize_word(line) == 0                  # 首字母大写单词占比高
            ):
                # 不要这个段落
                del_or_resv[idx] = 0  # 删除段落
                continue
            # else:
            #     ret_paragraphs.append(line)

        # 需要删除的短句之间的短句，满足规则也可删除。规则：1. 是短句；2. 不包含formula、ref，figure，table等token；3. 不在figure或table块内。
        # 情况一： 110... 第一个要删除的短句，前面为第1或第2或第3句
        # 情况二： ...0110... 要删除的短句之间的短句，连续不超过3句
        # 情况三： ...011 最后一个要删除的短句，后面时最后几句
        def _criterion(line_idx):
            if token_flag[line_idx] == 1:  # 属于 formula、ref、figure、table 或其区块
                return False
            if cls.paragraph_short(paragraphs[line_idx]) != 0:  # 不是短段落
                return False
            if cls._is_title(paragraphs[line_idx]):  # title 不删除
                return False
            return True

        # del_or_resv: 11011101
        # del_idxes  : 2,6
        del_idxes = np.where(np.array(del_or_resv) == 0)[0]
        for idx, del_idx in enumerate(del_idxes):  # del_idx: 要删除的短句的index，对应paragraphs
            if idx == 0:  # 这是第一个要删除的句子
                # del_idx 为2，循环0和1行
                if del_idx <= 3:  # 对应情况1
                    for chk_idx in range(0, del_idx):  # 看看第 0,1,2 句是否可以删除
                        if _criterion(chk_idx):  # 可以删除
                            del_or_resv[chk_idx] = 0
            else:
                # del_idx 为6，循环3,4,5行
                if del_idx - del_idxes[idx - 1] - 1 <= 3:  # 对应情况2
                    for chk_idx in range(del_idxes[idx - 1] + 1, del_idx):
                        if _criterion(chk_idx):  # 可以删除
                            del_or_resv[chk_idx] = 0

            if idx == len(del_idxes) - 1:  # 这是最后一个要删除的句子
                # del_idx 为6，循环第7行
                if len(paragraphs) - del_idx - 1 <= 3:  # 对应情况3
                    for chk_idx in range(del_idx + 1, len(paragraphs)):  # 看看最后3句是否可以删除
                        if _criterion(chk_idx):
                            del_or_resv[chk_idx] = 0

        ret_paragraphs = [line for line, flag in zip(paragraphs, del_or_resv) if flag == 1]
        return '\n'.join(ret_paragraphs)


if __name__ == "__main__":
    import os

    data_dir = "../data_100"
    data_nc = "../data_100_nc"
    os.makedirs(data_nc, exist_ok=True)
    files = os.listdir(data_dir)
    # files = ["01916122.2018.1443980.mmd"]

    for p in files:
        file_path = os.path.join(data_dir, p)
        ret = NonsenseCleaning.nonsense_cleaning(open(file_path, "r").read())
        open(os.path.join(data_nc, p), "w").write(ret)