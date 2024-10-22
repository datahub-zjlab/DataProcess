import logging
import os
import subprocess
import utils
import threading
import uuid
import time

class downloadThread(threading.Thread):
    def __init__(self, queue, dict, args):
        threading.Thread.__init__(self)
        self.dict = dict
        self.queue = queue
        self.args = args

    def run(self): 
        logging.info("Starting download thread")
        print("Starting download thread")
        for i in range(len(self.queue)):
            try:
                data = self.queue[i]
                if self.args.jobType != "crawler":
                    data_path, random_uuid = self.download_data(data)
                else:
                    data_path = ""
                    random_uuid = uuid.uuid4()
                # 单线程修改，多线程只读
                self.dict[data['id']] = [data_path, random_uuid]
                if i % 10 == 0:
                    print(f"have downlaod {i} files.")
                logging.info(f"{len(self.dict)} data download success!")
                print(f"{len(self.dict)} data download success!")
            except Exception as e:
                print(f"download error:{e}")
    # def download_data(self, data):
    #     random_uuid = uuid.uuid4()
    #     os.makedirs(str(random_uuid))
    #     local_path = str(random_uuid) + "/" + str(data["id"]) + ".pdf"
    #     if "geogpt-oss" in data["path"]:
    #         utils.proxy_yunqi_bucket.get_object_to_file(data["path"].replace("oss://geocloud/", "").replace("oss://geogpt-oss/", ""), local_path)
    #     else:
    #         utils.zhijiang_bucket.get_object_to_file(data["path"].replace("oss://geocloud/", "").replace("oss://geogpt-oss/", ""), local_path)
    #     return local_path, random_uuid

    def download_data(self, data):
        random_uuid = uuid.uuid4()
        os.makedirs(str(random_uuid))
        # filepath = data['path']
        # dir_name, file_name = os.path.split(filepath)
        # file_base, file_ext = os.path.splitext(file_name)
        if data['fileExtension'] is None:
            extension = '.pdf'
        else:
            extension = '.' + str(data['fileExtension']).lower()
        local_path = str(random_uuid) + "/" + str(data["id"]) + extension
        logging.info("data path is :" + data["path"].replace("oss://geocloud/", ""))
        oss_path = data["path"].replace("oss://geocloud/", "").replace("oss://geogpt-oss/", "")
        original_dir = "/app/"
        if "geogpt-oss" in data["path"]:
            utils.proxy_yunqi_bucket.get_object_to_file(oss_path, local_path)
        else:
            utils.zhijiang_bucket.get_object_to_file(oss_path, local_path)
        if extension == '.azw3':
            output_path = local_path.replace('.azw3', '.epub')
            try:
                command = ["ebook-converter", original_dir +local_path, original_dir +output_path]
                with open('./null', 'w') as devnull:
                        subprocess.run(command, stdout=devnull, stderr=subprocess.STDOUT, check=True)
            except Exception as e:
                print(e)
            return output_path, random_uuid

        if extension == '.djvu':
            output_path = local_path.replace('.djvu', '.pdf')
            try:
                convert_time = time.time()
                command = ["poetry", "run", "python", "-m", "dpsprep.dpsprep",original_dir+local_path, original_dir +output_path]
                # 使用 with 语句来确保文件正确关闭
                with open('./null', 'w') as devnull:
                #执行命令，并将标准输出和标准错误重定向到 /dev/null
                    subprocess.run(command, stdout=devnull, stderr=subprocess.STDOUT, check=True)
                cost_time = time.time() - convert_time
                print(f"convert {local_path} cost time :{cost_time}")
            except subprocess.CalledProcessError as e:
                print(f"Command failed with error: {e}")
            return output_path, random_uuid
        if extension not in ['.mobi','.epub','.pdf','.fb2','.svg','.txt','.xps','.cbz','.none']:
            raise Exception("file extension error!")
        return local_path, random_uuid
