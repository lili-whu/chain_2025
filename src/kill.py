import os
import re
import subprocess

def kill_process_by_port(port):
    # 使用netstat命令查找占用指定端口的进程
    command = f"netstat -ano | findstr :{port}"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)

    if result.returncode != 0:
        print(f"没有进程占用端口 {port}.")
        return

    # 获取进程ID（PID）
    lines = result.stdout.splitlines()
    for line in lines:
        match = re.search(r"(\d+)\s*$", line.strip())  # 正则提取PID
        if match:
            pid = match.group(1)
            print(f"找到占用端口 {port} 的进程PID: {pid}")

            # 使用taskkill命令终止该进程
            kill_command = f"taskkill /F /PID {pid}"
            kill_result = subprocess.run(kill_command, capture_output=True, text=True, shell=True)

            if kill_result.returncode == 0:
                print(f"成功终止进程PID {pid}.")
            else:
                print(f"终止进程PID {pid} 时发生错误: {kill_result.stderr}")
            return

    print(f"未能找到占用端口 {port} 的进程.")

if __name__ == "__main__":
    port_to_kill = 5000  # 设置要终止进程的端口号
    kill_process_by_port(port_to_kill)
    kill_process_by_port(port_to_kill)
    kill_process_by_port(port_to_kill)

