import argparse
import logging
import random

import requests
import time
import json
import queue
import threading

import utils
from processThread import processThread

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                    datefmt='%Y-%m-%d %A %H:%M:%S',
                    filename='data_process.log',
                    filemode='a')


parser = argparse.ArgumentParser()
parser.description = '参数列表详情'
parser.add_argument("--serviceURL", help="This is the URL of manage service", dest="serviceURL", type=str,
                    default="")
parser.add_argument("--jobID", help="This is the ID of job", dest="jobID", type=str,
                    default="")
parser.add_argument("--parseType", help="Parse type", dest="parseType", type=str,
                    default="")
parser.add_argument("--tagType", help="Tag type", dest="tagType", type=str,
                    default="")
parser.add_argument("--cleanType", help="Clean type", dest="cleanType", type=str,
                    default="")
parser.add_argument("--threadNum", help="threadNum", dest="threadNum", type=int,
                    default=1)
parser.add_argument("--outputPath", help="outputPath", dest="outputPath", type=str,
                    default="")
parser.add_argument("--jobType", help="job Type", dest="jobType", type=str,
                    default="")
parser.add_argument("--environment", help="running environment", dest="environment", type=str,
                    default="")
# parser.add_argument("--filter", help="filter", dest="filter", type=str,
#                     default="")
args = parser.parse_args()


def get_batch_task_wait(jobID, serviceURL):
    size = 0
    data_queue = None
    while True:
        try:
            params = {'jobId': jobID}
            logging.info("jobId: " + jobID)
            response = requests.get(serviceURL + "/taskDispatch/getBatchTask", params=params)
            if response.status_code != 200:
                logging.error(response.json())
                logging.error("Get batch task error.")
                time.sleep(random.randint(40, 80))
                continue
            print(json.loads(response.content))
            if json.loads(response.content)["code"] != 200:
                logging.error("Get batch task error.")
                time.sleep(random.randint(10, 20))
                continue
            data = json.loads(response.content)["data"]
            exist_tasks = data["existsTask"]
            if not exist_tasks:
                logging.info("No exist tasks.")
                return data_queue, size
            task_list = data["list"]
            size = len(task_list)
            data_queue = queue.Queue(maxsize=len(task_list))
            for task in task_list:
                data_queue.put(task)
            return data_queue, size
        except Exception as e:
            print(e)
            time.sleep(random.randint(40, 80))
            continue


def query_clickhouse(limit, offset):
    # 调用元数据服务或自行查询数据库，写入queue
    return queue.Queue(maxsize=limit)


def post_process_result(result_queue, args):
    data = []
    successTaskNum = 0
    failTaskNum = 0
    while not result_queue.empty():
        result = result_queue.get()
        if result[10]:
            successTaskNum += 1
        else:
            failTaskNum += 1
        data.append(result)
    client = utils.get_clickhouse_client(args)
    client.insert('process.data_process_result',
                  data, column_names=['ID', 'Version', 'DataID', 'DataDOI', 'Attribute1', 'Attribute2', 'ResultPath',
                                      'OriginPath', 'CreateDate', 'ModifyTime', 'IsSuccess', 'ParseType', 'TagType',
                                      'CleanType'])
    params = {'jobID': args.jobID, 'successTask': successTaskNum, 'failTask': failTaskNum}
    requests.post(args.serviceURL + "/DataPipeLine/PutPodResult", json=params)


def main():
    print(str(args))
    utils.set_bucket(args)
    utils.set_yunqi_bucket()
    utils.set_zhijiang_bucket()
    while True:
        logging.info("Start batch data process...")
        # 获取需要处理的书籍列表
        data_queue, size = get_batch_task_wait(args.jobID, args.serviceURL)
        #data_queue = query_clickhouse(limit, offset)
        if data_queue is None:
            break
        result_queue = queue.Queue(maxsize=size)
        logging.info("Start process...")
        # 多线程处理数据
        threads = []
        for i in range(args.threadNum):
            thread = processThread(i, data_queue, result_queue, args)
            thread.start()
            threads.append(thread)
        for i in threads:
            i.join()
        logging.info("Process end...")
        # 回调处理结果函数
        post_process_result(result_queue, args)


if __name__ == "__main__":
    import torch

    if torch.cuda.is_available():
        print("GPU is available.")
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            print(f"GPU {i}:")
            print(f"\tName: {torch.cuda.get_device_name(i)}")
            print(f"\tMemory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("No GPU found.")
    logging.info("Start data process agent...")
    main()
