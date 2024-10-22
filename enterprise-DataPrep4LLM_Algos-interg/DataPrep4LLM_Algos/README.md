
安装：

conda create -n ENV python=3.10

conda activate ENV

pip install -r requirements.txt

pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+pt2.3.1cu121


运行：

main.py --pdf ./test --lang en  
--pdf： 输入的pdf路径，可为单个pdf或文件夹。
--lang： 输入文件的语言，默认为中英文混杂，如果为纯英文文档，设为en的识别效果更好。

输出：

output/figures文件夹中包含所有插图，output/tables文件夹中包含所有表格，output/labels中包含所有图文对标注，output/<输入文件名>.md文件为pdf转成的markdown文档。
     
1. pip install -r requirements.txt
2. pip install ./detectron2-0.6+pt2.3.1cu121-cp310-cp310-linux_x86_64.whl 或从https://miropsota.github.io/torch_packages_builder/detectron2/ 下载合适版本安装。
3. vim /home/ubuntu/ENV_layout/lib/python3.10/site-packages/detectron2/data/transforms/transform.py 第46行Image.LINEAR修改为IMAGE.BILINEAR。
                                                                     