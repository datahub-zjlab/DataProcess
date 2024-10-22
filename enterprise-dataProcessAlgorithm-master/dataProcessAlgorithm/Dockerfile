from grobid/grobid:0.8.0
RUN mkdir -p /workspace
RUN echo "root:datapipeline" | chpasswd
RUN apt-get update
RUN apt-get install git expect libgl1-mesa-glx -y --force-yes
RUN pip install numpy clickhouse-connect pymupdf oss2 pdfplumber pandas frontend -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install grobid-client-python
RUN pip install opencv-python timm==0.5.4 python-Levenshtein albumentations pypdf orjson==3.3.0 PyPDF2 transformers==4.38.2 paddleocr paddlepaddle langid -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install fasttext langdetect -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN curl https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -o /usr/local/lib/python3.8/dist-packages/google/protobuf/internal/builder.py
COPY start.sh /workspace/
COPY start_yq.sh /workspace/
