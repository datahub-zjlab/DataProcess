#!/bin/bash

#cat > ~/.gitconfig << EOF
#[user]
#        name = liuyang2167@zhejianglab.com
#        password = datapipeline123
#[credential]
#        helper = store
#EOF

cd /workspace
expect <<EOF
    set timeout 10
    spawn git clone http://gitee.zhejianglab.com:80/enterprise/dataProcessAlgorithm.git
    expect "Username for*"
    send "liuyang2167@zhejianglab.org\n"
    expect "Password for*"
    send "datapipeline123\n"
    expect eof
EOF
cd dataProcessAlgorithm
git checkout master
mkdir -p static

# 如果parseType为"0"则后台拉起grobid进程
if [ "0" = "$3" ];then
    cd /opt/grobid/
    nohup /opt/grobid/grobid-service/bin/grobid-service &
    # sleep 30s等待grobid拉起
    sleep 30s
fi


cd /workspace/dataProcessAlgorithm
# 启动主进程
python main.py --serviceURL "$1" --jobID "$2" --parseType "$3" --tagType "$4" --cleanType "$5" --threadNum "$6" --outputPath "$7" --jobType "$8" --environment "$9"
