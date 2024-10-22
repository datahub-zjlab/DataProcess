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
    spawn git clone https://inner-gitlab.citybrain.org/gaochaolyf/dataprocessalgorithm.git
    expect "Username for*"
    send "gaochaolyf\n"
    expect "Password for*"
    send "Gc216001\n"
    expect eof
EOF
cd dataprocessalgorithm
git checkout master
mkdir -p static

# 如果parseType为"0"则后台拉起grobid进程
if [ "0" = "$3" ];then
    cd /opt/grobid/
    nohup /opt/grobid/grobid-service/bin/grobid-service &
    # sleep 30s等待grobid拉起
    sleep 30s
fi


cd /workspace/dataprocessalgorithm
# 启动主进程
python main.py --serviceURL "$1" --jobID "$2" --parseType "$3" --tagType "$4" --cleanType "$5" --threadNum "$6" --outputPath "$7" --jobType "$8" --environment "$9"
