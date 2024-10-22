# 使用官方的 Python 基础镜像
FROM python:3.10.12-slim-buster

# 设置工作目录
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir img2pdf \
    && pip install --no-cache-dir pikepdf \
    && pip install --no-cache-dir lxml
# 添加阿里云的pip镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码到容器中
# 复制 requirements.txt 并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY detectron2-0.6+pt2.3.1cu121-cp310-cp310-linux_x86_64.whl .
RUN pip install --no-cache-dir detectron2-0.6+pt2.3.1cu121-cp310-cp310-linux_x86_64.whl

# 先手动安装兼容的 opencv-python 版本
RUN pip install opencv-python==4.6.0.66
# 替换为 surya-ocr 需要的实际依赖项
RUN pip install filetype pydantic-settings pypdfium2 python-dotenv
# 使用阿里云的 pip 镜像源并安装项目所需的 Python 依赖
RUN pip install --no-cache-dir paddleocr==2.7.3 \
    && pip install --no-cache-dir surya-ocr==0.4.15 --no-deps transformers==4.42.4 \
    && pip install --no-cache-dir --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+pt2.3.1cu121 \
    && pip install --no-cache-dir pillow==9.1.0

# 设置为中国国内源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
# 安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    python3-dev \
    build-essential \
    git \
    expect \
    wget\
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install cnocr
RUN pip install clickhouse-connect oss2 pandas langid fasttext

# 安装 CuDNN
COPY cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz /tmp/


# 解压 CuDNN 文件
RUN mkdir -p /usr/local/cuda/include /usr/local/cuda/lib64 \
    && tar -xvf /tmp/cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz -C /tmp/ \
    && mv /tmp/cudnn-linux-x86_64-8.9.2.26_cuda12-archive/include/* /usr/local/cuda/include/ \
    && mv /tmp/cudnn-linux-x86_64-8.9.2.26_cuda12-archive/lib/* /usr/local/cuda/lib64/ \
    && rm -rf /tmp/cudnn-linux-x86_64-8.9.2.26_cuda12-archive*

# 设置环境变量
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc \
    && echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
#cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-4
# 设置环境变量
ENV PATH="/usr/local/cuda-12.4/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
# 复制代码到容器中
RUN rm -rf cudnn-linux-x86_64-8.9.2.26_cuda12-archive*
#安装dpsprep
# 更新 apt 源并安装基础工具
RUN echo "deb http://deb.debian.org/debian buster-backports main" >> /etc/apt/sources.list
RUN apt-get -y install curl djvulibre-bin libdjvulibre-dev pkg-config
RUN pip install djvulibre-python

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"
RUN git clone https://github.com/kcroker/dpsprep.git /app/dpsprep
# 复制方便的脚本
RUN cp /app/dpsprep/pyproject.toml .
RUN poetry install
# 安装必要的依赖库
RUN apt-get install -y \
    libxml2-dev \
    && export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig
# 克隆并进入项目目录
RUN git clone https://github.com/gryf/ebook-converter /app/ebook
# 安装依赖
RUN pip install -r ebook/requirements.txt ebook/.
RUN apt-get -y install vim
RUN pip install scikit-learn
COPY . .
# 执行 ldconfig 以更新共享库缓存
RUN ldconfig
