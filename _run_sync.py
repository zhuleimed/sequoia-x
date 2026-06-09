"""手动启动同步并监控进度的辅助脚本"""
import paramiko
import time
import sys

HOST = "2001:250:4400:89:aae6:63d8:a8e0:51dc"
PORT = 22
USER = "zhulei"
PWD = "zhulei@HPC88660159"
RD = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x"
PY = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"
LOG = RD + "/logs/sync_manual_20260609.log"


def run_cmd(client, cmd):
    stdin, stdout, stderr = client.exec_command(cmd)
    return stdout.read().decode(), stderr.read().decode()


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=10)

# 清旧日志
run_cmd(client, "rm -f " + LOG)

# 启动同步
cmd = "cd " + RD + " && nohup " + PY + " main.py --sync-only > " + LOG + " 2>&1 & echo PID=$!"
out, err = run_cmd(client, cmd)
pid = out.strip()
print("启动PID:", pid)

# 监控
for i in range(60):
    time.sleep(10)
    out2, _ = run_cmd(client, "tail -5 " + LOG + " 2>/dev/null")
    lines = out2.strip().split("\n")

    pid_val = pid.split("=")[-1].strip() if "=" in pid else pid.strip()
    out3, _ = run_cmd(client, "ps -p " + pid_val + " > /dev/null 2>&1 && echo alive || echo dead")
    status = out3.strip()

    last_line = lines[-1][:120] if lines and lines[0] else "无日志"
    print(f"+{i*10+10:3d}s | 进程:{status:5s} | {last_line}")

    if status == "dead":
        print("\n=== 完整日志 ===")
        out4, _ = run_cmd(client, "cat " + LOG)
        print(out4[:4000])
        break

client.close()
