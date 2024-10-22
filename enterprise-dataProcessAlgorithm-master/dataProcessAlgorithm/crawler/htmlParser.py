#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @File     htmlParser.py.py
    @Author   shilixin
    @Date     2024/7/5 9:25
    @Describe 
    @Version  1.0
"""
import logging
import os
import json
import fasttext
from bs4 import BeautifulSoup
import re
from langdetect import detect, DetectorFactory

import utils


def clean_text(text):
    # 去除多余的空白符
    text = re.sub(r' +', ' ', text)
    # 去除多余空白行
    text = re.sub(r' *[\r\n]+', '\n', text)
    text = re.sub(r"\n+ *", '\n', text)
    return text.strip()


def lang_detect(text):
    # 设置随机种子，确保结果的一致性
    DetectorFactory.seed = 0

    try:
        language = detect(text)
    except Exception as e:
        language = ""
        logging.error("Error:" + str(e))
    return language


# :TODO 学科检测
def subject_detect(text):
    model = fasttext.load_model("/workspace/subject_textclf/classifier_multi_subject.bin")
    subject = utils.get_subject(text, model)
    return subject


def parse_html_files(input_dir, output_dir, data, random_uuid):
    # 单个文件和合并后文件，相对路径的比较路径
    relpath_start = os.getcwd()
    results = []
    results_merge = []
    # 创建输出目录，如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                contents = file.read()

            soup = BeautifulSoup(contents, 'lxml')

            # 移除style和script标签
            for style in soup(['style', 'script']):
                style.extract()

            # 获取文本，使用空格作为分隔符，并保留首尾空白
            texts = soup.get_text(separator=' ', strip=False)
            # 清洗文本
            cleaned_text = clean_text(texts)
            # 检测语种
            language = lang_detect(cleaned_text)
            # 检测学科
            subject = subject_detect(cleaned_text)

            # 返回参数信息
            upload_file_path = "basic/arxiv/" + data["version"] + "/" + str(random_uuid) + "/" + filename
            utils.upload_file(upload_file_path, filepath)
            result_entry = {
                "url": filename.replace('.html', ''),
                "html_file_path": upload_file_path,
                "language": language,
                "subject": subject
            }
            results.append(result_entry)

            # 合并信息
            result_merge_entry = {
                "text": cleaned_text,
                "url": filename.replace('.html', ''),
                "html_file_path": upload_file_path,
                "language": language,
                "subject": subject
            }
            results_merge.append(result_merge_entry)

            # 写入结果到新文件
            with open(output_filepath, 'w', encoding='utf-8') as out_file:
                out_file.write(cleaned_text)
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")
            return False, "", str(e), ""

    # 合并所有结果到一个JSON文件
    merge_output_path = os.path.join(output_dir, 'merged_results.json')
    with open(merge_output_path, 'w', encoding='utf-8') as merge_file:
        json.dump(results_merge, merge_file, ensure_ascii=False, indent=4)
    merge_relative_output_filepath = os.path.relpath(merge_output_path, start=relpath_start)

    return True, merge_relative_output_filepath, "", str(results)


if __name__ == "__main__":
    input_directory = r'E:\2024-07-03-dataPipeline\workspace\dataProcess\www.drdouglaschristie.com'
    output_directory = r'E:\2024-07-03-dataPipeline\workspace\dataProcess\output'
    relative_output_filepath, results = parse_html_files(input_directory, output_directory)
    logging.info(relative_output_filepath, results)
