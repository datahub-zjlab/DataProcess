import logging
import os
import time

import utils
from data_process import data_process
import threading
import uuid
import oss2
import shutil
from paddleocrmix import paddleOCRMix
from type_enum import parse_type_enum

namespace = {"": 'http://www.tei-c.org/ns/1.0'}

class processThread(threading.Thread):
    def __init__(self, threadID, queue, result_queue, data_dict, args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.queue = queue
        self.result_queue = result_queue
        self.data_dict = data_dict
        self.args = args

    def run(self):
        logging.info("Starting threadID : " + str(self.threadID))
        print("Starting threadID : " + str(self.threadID))
        # 初始化模型
        pdom_model = None
        if self.args.parseType == parse_type_enum.paddleOCRMix.value:
            pdom_model = paddleOCRMix()
            pdom_model.model_init()
        while True:
            # 从待处理队列中获取要处理的元数据
            if self.queue is None or self.queue.empty():
                break
            else:
                data = self.queue.get(block=False)
            # 下载要处理的数据
            if self.args.jobType != "crawler":
                data_path, random_uuid = self.data_dict.get(data['id'],[None,None])
                retry_time = 3
                while data_path == None and random_uuid ==None and retry_time>0:
                    time.sleep(10)
                    data_path, random_uuid = self.data_dict.get(data['id'],[None,None])
                    retry_time -= 1
                if data_path == None and random_uuid ==None :
                    print(f"{data} process fail.data_dict value not exist.")
                    self.result_queue.put([str(random_uuid), data["version"], data["id"], data["doi"], f"{data} process fail.data_dict value not exist.",
                                           "", "", data["path"], str(round(time.time() * 1000)),
                                           str(round(time.time() * 1000)), False, self.args.parseType,
                                           self.args.tagType, self.args.cleanType])
                    continue
            else:
                data_path = ""
                random_uuid = uuid.uuid4()
            process = data_process(self.result_queue, self.args, data_path, data, str(random_uuid),pdom_model)
            if self.args.jobType == "pdfParse":
                is_success, data_path, message, properties = process.run_pdfParse()
            elif self.args.jobType == "crawler":
                is_success, data_path, message, properties = process.run_crawler()
            elif self.args.jobType == "cc":
                is_success, data_path, message, properties = process.run_cc()
            else:
                is_success, data_path, message, properties = False, "", "Wrong jobType.", ""
            upload_file_url = ""
            if is_success:
                upload_file_url = ("pipeline_result/algos_version_" + str(utils.algos_version) + "/" + data["version"] + "/" + str(
                    random_uuid) + "/" + str(random_uuid) + ".mmd")
                utils.upload_file(upload_file_url, data_path)
            self.result_queue.put([str(random_uuid), data["version"], data["id"], data["doi"], message,
                                   properties, upload_file_url, data["path"], str(round(time.time() * 1000)),
                                   str(round(time.time() * 1000)), is_success, self.args.parseType,
                                   self.args.tagType, self.args.cleanType])
            self.delete_data(random_uuid)



    def delete_data(self, random_uuid):
        try:
            shutil.rmtree(str(random_uuid), ignore_errors=True)
            shutil.rmtree(str(random_uuid) + "_output", ignore_errors=True)
            shutil.rmtree(str(random_uuid) + "_png", ignore_errors=True)
        except:
            pass
        return

    def del_temp_files(self, paths):
        try:
            """删除临时文件"""
            for file in paths:
                os.remove(file)
        except Exception:
            pass
