from enum import Enum


class parse_type_enum(Enum):
    grobid = "0"
    nougat = "1"


class tag_type_enum(Enum):
    paddleOCR = "0"


class clean_type_enum(Enum):
    dedup = "0"
