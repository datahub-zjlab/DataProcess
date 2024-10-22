import logging
import os
import time

import utils
from data_process import data_process
import threading
import uuid
import oss2
import shutil

namespace = {"": 'http://www.tei-c.org/ns/1.0'}

class processThread(threading.Thread):
    def __init__(self, threadID, queue, result_queue, args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.queue = queue
        self.result_queue = result_queue
        self.args = args

    def run(self):
        logging.info("Starting threadID : " + str(self.threadID))
        while True:
            # 从待处理队列中获取要处理的元数据
            if self.queue is None or self.queue.empty():
                break
            else:
                data = self.queue.get(block=False)
            # 下载要处理的数据
            if self.args.jobType != "crawler":
                data_path, random_uuid = self.download_data(data)
            else:
                data_path = ""
                random_uuid = uuid.uuid4()
            process = data_process(self.result_queue, self.args, data_path, data, str(random_uuid))
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
                upload_file_url = ("basic/arxiv/" + data["version"] + "/" + str(random_uuid)
                                   + "/" + str(random_uuid) + ".mmd")
                utils.upload_file(upload_file_url, data_path)
            self.result_queue.put([str(random_uuid), data["version"], data["id"], data["doi"], message,
                                   properties, upload_file_url, data["path"], str(round(time.time() * 1000)),
                                   str(round(time.time() * 1000)), is_success, self.args.parseType,
                                   self.args.tagType, self.args.cleanType])
            self.delete_data(random_uuid)

    def download_data(self, data):
        random_uuid = uuid.uuid4()
        os.makedirs(str(random_uuid))
        local_path = str(random_uuid) + "/" + str(data["id"]) + ".pdf"
        if "geogpt-oss" in data["path"]:
            utils.proxy_yunqi_bucket.get_object_to_file(data["path"].replace("oss://geocloud/", "").replace("oss://geogpt-oss/", ""), local_path)
        else:
            utils.zhijiang_bucket.get_object_to_file(data["path"].replace("oss://geocloud/", "").replace("oss://geogpt-oss/", ""), local_path)
        return local_path, random_uuid

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
