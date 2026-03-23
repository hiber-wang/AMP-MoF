#!/bin/bash

# 定义需要执行的命令
COMMANDS=(
    "python main.py --watch --agent vgg16 --record"
)

# main.py程序是否在运行中
check_process() {
    pgrep -f main.py > /dev/null
}

# 执行命令函数
execute_commands() {
    while true; do
        check_process
        if [ $? -ne 0 ]; then
            echo "main.py程序意外终止, 等待10秒后重新执行命令..."
            for cmd in "${COMMANDS[@]}"; do
                echo "Executing command: ${cmd#sudo}"
                sleep 1
                if [[ $cmd == sudo* ]]; then
                    echo "001002" | sudo -S sh -c "${cmd#sudo}"
                else
                    eval "$cmd"
                fi
                # main.py程序是否在运行中
            done
        fi
        sleep 1
    done
}

# 执行命令
execute_commands
