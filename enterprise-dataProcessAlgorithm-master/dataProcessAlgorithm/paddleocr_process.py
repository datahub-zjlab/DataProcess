# 要求：输入pdf路径，data，random_uuid
# 输出：字典{'table':[{'location': '表1的位置坐标，包括在第几页'， 'oss_path': '表1截图的oss路径'}, {'location': '表2的位置坐标，包括在第几页'， 'oss_path': '表2截图的oss路径'}], 'figure':[{'location': '图1的位置坐标，包括在第几页'， 'oss_path': '图1截图的oss路径'}, {'location': '图2的位置坐标，包括在第几页'， 'oss_path': '图2截图的oss路径'}]}
# 其中oss_path: OSS上传路径：upload_file_url = ("basic/arxiv/" + data["version"] + "/" + str(random_uuid) + "/" + "1.jpg")

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os
import cv2
from paddleocr import PPStructure, save_structure_res
import fitz

import utils


def pdf_to_images(pdf_path, output_folder):
    # 打开PDF文档
    doc = fitz.open(pdf_path)
    try:
        os.makedirs(output_folder, exist_ok=True)
    except FileExistsError:
        pass
    # 遍历每一页
    for page_number in range(doc.page_count):
        # 获取指定页面
        page = doc.load_page(page_number)
        # 将页面渲染为图像
        pix = page.get_pixmap(dpi=300)
        # 保存图像到文件
        # image_file_name = f"{output_folder}//page_{page_number + 1}.png"
        image_file_name = output_folder + "//page" + str(int(page_number + 1)) + '.png'
        pix.save(image_file_name)


def paddle_process(pdf_path, data, random_uuid):
    keys = ['table', 'figure', 'version']
    default_value = None
    res_dict = {key: default_value for key in keys}

    pdf2img_path = os.path.dirname(pdf_path) + "_png//" + pdf_path.split("//")[-1].replace(".pdf", "")
    pdf_to_images(pdf_path, pdf2img_path)

    table_engine = PPStructure(table=False, ocr=False, show_log=True)
    save_folder = os.path.dirname(pdf_path) + "_output//" + pdf_path.split("//")[-1].replace(".pdf", "")
    try:
        os.makedirs(save_folder, exist_ok=True)
    except FileExistsError:
        pass

    upload_file = "basic/arxiv/" + data["version"] + "/" + str(random_uuid) + "/"
    table_list = []
    figure_list = []
    table_cnt = 0
    figure_cnt = 0
    for filename in os.listdir(pdf2img_path):

        if filename.endswith(('.png', '.jpg', '.jpeg', 'bmp')):
            file_path = os.path.join(pdf2img_path, filename)

            img = cv2.imread(file_path)
            result = table_engine(img)
            image = Image.open(file_path)

            for line in result:
                line.pop('img')
                if line['type'] == "figure":
                    figure_cnt += 1
                    cropped_img = image.crop(line['bbox'])
                    cropped_img.save(save_folder + "//figure_" + str(line['bbox']) + ".png")
                    upload_file_url = upload_file + "figure" + str(figure_cnt) + ".png"
                    utils.upload_file(upload_file_url, save_folder + "//figure_" + str(line['bbox']) + ".png")
                    figure_list.append({'location': [line['bbox'], int(filename[4:].replace(".png", ""))],
                                        'oss_path': upload_file_url})

                if line['type'] == "table":
                    table_cnt += 1
                    cropped_img = image.crop(line['bbox'])
                    cropped_img.save(save_folder + "//table_" + str(line['bbox']) + ".png")
                    upload_file_url = upload_file + "table" + str(table_cnt) + ".png"
                    utils.upload_file(upload_file_url, save_folder + "//table_" + str(line['bbox']) + ".png")
                    table_list.append({'location': [line['bbox'], int(filename[4:].replace(".png", ""))],
                                       'oss_path': upload_file_url})
    res_dict['figure'] = figure_list
    res_dict['table'] = table_list
    res_dict['version'] = 3
    return res_dict
