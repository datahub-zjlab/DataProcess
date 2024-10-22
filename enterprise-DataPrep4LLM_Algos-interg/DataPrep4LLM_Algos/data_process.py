# import crawler.crawler_process
from type_enum import parse_type_enum, tag_type_enum, clean_type_enum


# from grobid import grobid_process
# from nougat import nougat_process

class data_process(object):
    def __init__(self, result_queue, args, data_path, data, random_uuid, model):
        self.result_queue = result_queue
        self.args = args
        self.data_path = data_path
        self.data = data
        self.result_path = ""
        self.random_uuid = random_uuid
        self.process_model = model

    # def grobid(self):
    #     return grobid_process.main(self.data_path, self.data, self.random_uuid)
    #
    # def nougat(self):
    #     return nougat_process.main(self.data_path, self.data, self.random_uuid)

    def paddleOCRMix(self):
        return self.process_model.process(self.data_path, self.data, self.random_uuid)

    def paddleOCR(self):
        pass

    def dedup(self):
        pass

    def run_pdfParse(self):
        message = ""
        properties = ""
        # 解析参数
        parseType = self.args.parseType
        print("parseType:"+parseType)
        for i in parse_type_enum:
            if parseType == i.value:
                func = getattr(self, i.name)
                properties, is_success, self.data_path, message = func()
                if not is_success:
                    return False, "", message, properties

        tag_type_list = self.args.tagType.split(",")
        for i in tag_type_enum:
            if i.value in tag_type_list:
                func = getattr(self, i.name)
                is_success, self.data_path, message = func()
                if not is_success:
                    return False, "", message, properties

        clean_type_list = self.args.cleanType.split(",")
        for i in clean_type_enum:
            if i.value in clean_type_list:
                func = getattr(self, i.name)
                is_success, self.data_path, message = func()
                if not is_success:
                    return False, "", message, properties
        return True, self.data_path, message, properties

    def run_crawler(self):
        pass
        # return crawler.crawler_process.main(self.data, self.random_uuid)

    def run_cc(self):
        pass
