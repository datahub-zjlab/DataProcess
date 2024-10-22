import logging
import traceback

import clickhouse_connect
import oss2

import utils

auth = None
bucket = None
proxy_yunqi_auth = None
proxy_yunqi_bucket = None
zhijiang_auth = None
zhijiang_bucket = None
algos_version = 3

def read_data(data_path):
    with open(data_path) as file_obj:
        contents = file_obj.read()
    return contents


def save_data(text, data_path):
    with open(data_path, 'w+') as f:
        f.write(text)
    return


def upload_file(file_url, local_path):
    try:
        utils.bucket.put_object_from_file(file_url, local_path)
        return True
    except Exception as e:
        logging.error(e)
        traceback.print_exc()


def get_subject(text, model):
    text = str(text).replace('\n', '')
    result = model.predict(text)
    result = result[0][0]
    return result


def get_clickhouse_client(args):
    if args.environment == "zhijiang":
        client = clickhouse_connect.get_client(
            host='10.200.48.74',
            port=8123,
            username='zjlab',
            password='zjlab',
            database='process'
        )
    else:
        client = clickhouse_connect.get_client(
            host='172.27.213.44',
            port=31238,
            username='zjlab',
            password='zjlab',
            database='process'
        )
    return client


def set_bucket(args):
    if args.environment == "zhijiang":
        utils.auth = oss2.Auth('kJscloSzed09Lhy7', 'mQyqefxOLd7SPUPKiTam3JYsHhut12')
        utils.bucket = oss2.Bucket(utils.auth, 'http://oss-cn-hangzhou-zjy-d01-a.ops.cloud.zhejianglab.com/', 'geocloud')
    else:
        utils.auth = oss2.Auth('oIvwiz94CqpIGH5y', 'GGTyHxhoE05SccPrc2TDMJwIrR3zGX')
        utils.bucket = oss2.Bucket(utils.auth, 'http://oss-cn-jswx-xuelang-d01-a.ops.cloud.wuxi-yqgcy.cn/', 'geogpt-oss')


def set_yunqi_bucket():
    utils.proxy_yunqi_auth = oss2.Auth('oIvwiz94CqpIGH5y', 'GGTyHxhoE05SccPrc2TDMJwIrR3zGX')
    utils.proxy_yunqi_bucket = oss2.Bucket(utils.proxy_yunqi_auth, 'http://172.27.213.44:8080/', 'geogpt-oss')


def set_zhijiang_bucket():
    utils.zhijiang_auth = oss2.Auth('kJscloSzed09Lhy7', 'mQyqefxOLd7SPUPKiTam3JYsHhut12')
    utils.zhijiang_bucket = oss2.Bucket(utils.zhijiang_auth, 'http://oss-cn-hangzhou-zjy-d01-a.ops.cloud.zhejianglab.com/', 'geocloud')

