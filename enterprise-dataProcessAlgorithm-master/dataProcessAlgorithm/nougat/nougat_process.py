"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
from pathlib import Path
import logging
import re
import argparse
import re
import os
from functools import partial
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device, default_batch_size
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
import pypdf

import shutil
import PyPDF2
from nougat.nougatpost import nougat_post
import paddleocr_process

logging.basicConfig(level=logging.INFO)


class args(object):
    def __init__(self, batchsize, checkpoint, model, out, recompute, full_precision, no_markdown, markdown, skipping,
                 no_skipping, pages, pdf):
        self.batchsize = batchsize
        self.checkpoint = checkpoint
        self.model = model
        self.out = out
        self.recompute = recompute
        self.full_precision = full_precision
        self.no_markdown = no_markdown
        self.markdown = markdown
        self.skipping = skipping
        self.no_skipping = no_skipping
        self.pages = pages
        self.pdf = pdf

        if self.checkpoint is None or not self.checkpoint.exists():
            self.checkpoint = get_checkpoint(self.checkpoint, model_tag=self.model)
        if self.out is None:
            logging.warning("No output directory. Output will be printed to console.")
        else:
            if not os.path.exists(self.out):
                logging.info("Output directory does not exist. Creating output directory.")
                try:
                    os.makedirs(self.out, exist_ok=True)
                except FileExistsError:
                    pass
            if not os.path.isdir(self.out):
                logging.error("Output has to be directory.")
                sys.exit(1)
        print("self.pdf = ", self.pdf)
        '''
        if len(self.pdf) == 1 and not self.pdf[0].suffix == ".pdf":
            # input is a list of pdfs
            try:
                pdfs_path = self.pdf[0]
                if pdfs_path.is_dir():
                    self.pdf = list(pdfs_path.rglob("*.pdf"))
                else:
                    self.pdf = [
                        Path(l) for l in open(pdfs_path).read().split("\n") if len(l) > 0
                    ]
                logging.info(f"Found {len(self.pdf)} files.")
            except:
                pass
        if self.pages and len(self.pdf) == 1:
            pages = []
            for p in self.pages.split(","):
                if "-" in p:
                    start, end = p.split("-")
                    pages.extend(range(int(start) - 1, int(end)))
                else:
                    pages.append(int(p) - 1)
            self.pages = pages
        else:
            self.pages = None
        '''


def copy_files_by_name_pattern(missing_page_list, source_dir, target_dir, pattern):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in missing_page_list:
        if filename.endswith(pattern):
            # 构建完整的文件路径
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)

            # 复制文件
            shutil.copy2(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")


def extract_pages_from_pdf(input_pdf_path, output_pdf_path, page_numbers):
    with open(input_pdf_path, 'rb') as input_file:
        pdf_reader = PyPDF2.PdfReader(input_file)
        pdf_writer = PyPDF2.PdfWriter()

        for page_num in page_numbers:
            page = pdf_reader.pages[page_num - 1]  # page_numbers should be 1-indexed
            pdf_writer.add_page(page)

        with open(output_pdf_path, 'wb') as output_file:
            pdf_writer.write(output_file)


def small(pdf_path, data, random_uuid):
    # args = get_args()  # 原来获取args的方法，现在改为类
    args_small = args(
        batchsize=1,
        checkpoint=None,
        model="0.1.0-small",
        out="output",
        recompute=True,
        full_precision=False,
        no_markdown=False,
        markdown=True,
        skipping=True,
        no_skipping=False,
        pages=None,
        pdf=pdf_path)

    logging.info(data["path"] + " process start.")
    if not os.path.exists(pdf_path):
        raise Exception(pdf_path + " not found.")
    logging.info("run nougat......")

    id = str(data["id"])
    # result_mmd_path = pdf_path.replace(id + ".pdf", str(random_uuid) + "_" + id + ".mmd")
    result_mmd_path = str(random_uuid) + "/" + str(random_uuid) + "_" + id + ".mmd"

    model = NougatModel.from_pretrained(args_small.checkpoint, ignore_mismatched_sizes=True)
    model = move_to_device(model, bf16=not args_small.full_precision, cuda=args_small.batchsize > 0)
    if args_small.batchsize <= 0:
        # set batch size to 1. Need to check if there are benefits for CPU conversion for >1
        args_small.batchsize = 1
    model.eval()
    datasets = []
    for pdf in [args_small.pdf]:
        if not os.path.exists(pdf):
            continue
        if args_small.out:
            out_path = Path(args_small.out) / Path(pdf).with_suffix(".mmd").name
            if os.path.exists(out_path) and not args_small.recompute:
                logging.info(
                    f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                )
                continue
        try:
            dataset = LazyDataset(
                str(pdf),
                partial(model.encoder.prepare_input, random_padding=False),
                args_small.pages,
            )
        except pypdf.errors.PdfStreamError:
            logging.info(f"Could not load file {str(pdf)}.")
            continue
        datasets.append(dataset)
    if len(datasets) == 0:
        return
    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(datasets),
        batch_size=args_small.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    predictions = []
    file_index = 0
    page_num = 0

    missing_page_list = []
    missing_page_num = []

    for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
        model_output = model.inference(
            image_tensors=sample, early_stopping=args_small.skipping
        )
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
            if page_num == 0:
                logging.info(
                    "Processing file %s with %i pages"
                    % (datasets[file_index].name, datasets[file_index].size)
                )
                print("datasets[file_index].name=", datasets[file_index].name)
                print("type datasets[file_index].name=", type(datasets[file_index].name))
            page_num += 1

            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                logging.warning(f"reason 01 for {datasets[file_index].name}")
                print("reason 01")
                missing_page_list.append(datasets[file_index].name.split("/")[1])
                missing_page_num.append(page_num)
            elif args_small.skipping and model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    logging.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                    logging.warning(f"reason 02 for {datasets[file_index].name}")
                    print("reason 02")
                    missing_page_list.append(datasets[file_index].name.split("/")[1])
                    missing_page_num.append(page_num)

                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i * args_small.batchsize + j + 1}]\n\n"
                    )
                    logging.warning(f"reason 03 for {datasets[file_index].name}")
                    print("reason 03")
                    missing_page_list.append(datasets[file_index].name.split('/')[1])
                    missing_page_num.append(page_num)
            else:
                if args_small.markdown:
                    output = markdown_compatible(output)

                predictions.append(output)
            if is_last_page[j]:
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                if args_small.out:
                    print("get small model output")
                    # out_path = args.out / Path(is_last_page[j]).with_suffix(".mmd").name
                    # out_path.parent.mkdir(parents=True, exist_ok=True)
                    # out_path.write_text(out, encoding="utf-8")
                else:
                    print(out, "\n\n")
                # predictions = []
                page_num = 0
                file_index += 1
    # input_pdf_path = str(args_small.pdf[0])
    print("args_small.pdf = ", args_small.pdf)
    input_pdf_path = str(args_small.pdf)

    output_pdf_path = input_pdf_path.split(".")[0] + "_mp.pdf"
    print("output_pdf_path = ", output_pdf_path)
    print("missing_page_num=", missing_page_num)
    extract_pages_from_pdf(input_pdf_path, output_pdf_path, missing_page_num)

    return output_pdf_path, predictions, missing_page_num, result_mmd_path


def base(output_pdf_path, predictions, missing_page_num, result_mmd_path):
    try:
        # args = get_args()
        args_base = args(
            batchsize=1,
            checkpoint=None,
            model="0.1.0-base",
            out="output",
            recompute=True,
            full_precision=False,
            no_markdown=False,
            markdown=True,
            skipping=True,
            no_skipping=False,
            pages=None,
            pdf=Path(output_pdf_path))

        args_base.checkpoint = Path(str(args_base.checkpoint.parent) + '/nougat-0.1.0-base')
        model = NougatModel.from_pretrained(args_base.checkpoint, ignore_mismatched_sizes=True)
        model = move_to_device(model, bf16=not args_base.full_precision, cuda=args_base.batchsize > 0)
        if args_base.batchsize <= 0:
            # set batch size to 1. Need to check if there are benefits for CPU conversion for >1
            args_base.batchsize = 1
        model.eval()
        datasets = []
        for pdf in [args_base.pdf]:
            if not os.path.exists(pdf):
                continue
            if args_base.out:
                out_path = Path(args_base.out) / Path(pdf).with_suffix(".mmd").name
                if os.path.exists(out_path) and not args_base.recompute:
                    logging.info(
                        f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                    )
                    continue
            try:
                dataset = LazyDataset(
                    str(pdf),
                    partial(model.encoder.prepare_input, random_padding=False),
                    args_base.pages,
                )
            except pypdf.errors.PdfStreamError:
                logging.info(f"Could not load file {str(pdf)}.")
                continue
            datasets.append(dataset)
        if len(datasets) == 0:
            return
        dataloader = torch.utils.data.DataLoader(
            ConcatDataset(datasets),
            batch_size=args_base.batchsize,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )

        # predictions = []
        file_index = 0
        page_num = 0

        missing_page_list = []
        mpn_index = 0

        for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
            model_output = model.inference(
                image_tensors=sample, early_stopping=args_base.skipping
            )
            # check if model output is faulty
            for j, output in enumerate(model_output["predictions"]):
                if page_num == 0:
                    logging.info(
                        "Processing file %s with %i pages"
                        % (datasets[file_index].name, datasets[file_index].size)
                    )
                page_num += 1

                if output.strip() == "[MISSING_PAGE_POST]":
                    # uncaught repetitions -- most likely empty page
                    # predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                    logging.warning(f"reason 01 for {datasets[file_index].name}")
                    print("reason 01")
                    missing_page_list.append(datasets[file_index].name.split("/")[1])

                elif args_base.skipping and model_output["repeats"][j] is not None:
                    if model_output["repeats"][j] > 0:
                        # If we end up here, it means the output is most likely not complete and was truncated.
                        logging.warning(f"Skipping page {page_num} due to repetitions.")
                        # predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                        logging.warning(f"reason 02 for {datasets[file_index].name}")
                        print("reason 02")
                        missing_page_list.append(datasets[file_index].name.split("/")[1])


                    else:
                        # If we end up here, it means the document page is too different from the training domain.
                        # This can happen e.g. for cover pages.
                        # predictions.append(
                        #     f"\n\n[MISSING_PAGE_EMPTY:{i * args.batchsize + j + 1}]\n\n"
                        # )
                        logging.warning(f"reason 03 for {datasets[file_index].name}")
                        print("reason 03")
                        # missing_page_list.append(datasets[file_index].name.split('/')[1])
                else:
                    if args_base.markdown:
                        output = markdown_compatible(output)
                    predictions[missing_page_num[mpn_index] - 1] = output
                mpn_index += 1
                if is_last_page[j]:
                    out = "".join(predictions).strip()
                    out = re.sub(r"\n{3,}", "\n\n", out).strip()
                    if args_base.out:
                        # out_path = args_base.out / Path(is_last_page[j]).with_suffix(".mmd").name
                        out_path = result_mmd_path
                        # out_path.parent.mkdir(parents=True, exist_ok=True)
                        # try:
                        #     os.makedirs(out_path, exist_ok=True)
                        # except FileExistsError:
                        #     pass
                        Path(out_path).write_text(out, encoding="utf-8")
                    else:
                        print(out, "\n\n")
                    predictions = []
                    page_num = 0
                    file_index += 1
        os.remove(output_pdf_path)
        return True, result_mmd_path, "success"
    except Exception as e:
        logging.error(output_pdf_path.replace("_mp.pdf", ".pdf") + " process failed. Error: " + str(e))
        logging.exception(e)
        return False, "", str(e)


def main(pdfpath, data, random_uuid):
    print("pdfpath = ", pdfpath)
    print("data = ", data)
    print("random_uuid = ", random_uuid)
    paddleocr_dict = paddleocr_process.paddle_process(pdfpath, data, random_uuid)
    out_path, pre, missing, result_mmd_path = small(pdfpath, data, random_uuid)
    is_success, result_path, message = base(out_path, pre, missing, result_mmd_path)
    if is_success:
        post_success, post_message, subject = nougat_post(result_path)
        is_success &= post_success
        message += post_message
        paddleocr_dict['subject'] = subject
    return str(paddleocr_dict), is_success, result_path, message
