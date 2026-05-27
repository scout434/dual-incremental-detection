

"""
从步骤.txt文件读取命令并执行，同时记录日志到outputs/log目录
"""

import subprocess
import sys
import re
import os
from datetime import datetime
from pathlib import Path


def strip_ansi_codes(text: str) -> str:
    """移除 ANSI 转义序列（颜色码）"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def run_commands_from_file(steps_file: str, log_dir: str = "outputs/log"):
    """
    从文本文件读取命令并执行，每条命令的输出都会记录到单独的日志文件中

    Args:
        steps_file: 包含命令的步骤文件路径
        log_dir: 日志存储目录
    """
    steps_path = Path(steps_file)
    if not steps_path.exists():
        print(f"错误：步骤文件 {steps_file} 不存在")
        return

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 读取步骤文件
    with open(steps_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 过滤空行和注释行
    commands = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            commands.append(line)

    if not commands:
        print(f"警告：步骤文件 {steps_file} 中没有找到有效命令")
        return

    print(f"从 {steps_file} 中读取到 {len(commands)} 条命令")
    print("=" * 50)

    # 执行每条命令
    for i, command in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] 执行命令: {command}")
        print("-" * 50)

        # 为每条命令生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_command = "".join(c if c.isalnum() or c in {"-", "_", " "} else "_" for c in command)
        safe_command = safe_command[:50]  # 限制文件名长度
        log_file = log_path / f"step_{i}_{timestamp}_{safe_command}.log"

        # 构建带日志记录的命令
        # 使用 subprocess 执行命令，并将输出重定向到日志文件
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                # 写入命令信息
                f.write(f"命令: {command}\n")
                f.write(f"开始时间: {datetime.now()}\n")
                f.write("=" * 50 + "\n")

                # 【修改点 1】设置环境变量禁用颜色输出，并强制子进程的 Python 内部也使用 UTF-8
                env = os.environ.copy()
                env['NO_COLOR'] = '1'
                env['ULTRALYTICS_COLOR'] = '0'
                env['PYTHONIOENCODING'] = 'utf-8'  # 🌟 强制子进程的 Python 使用 UTF-8 编码

                # 执行命令
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',       # 🌟【修改点 2】显式指定使用 utf-8 编码解码子进程流
                    errors='ignore',        # 🌟【修改点 3】如果遇到极其顽固的乱码字节直接跳过，防止程序崩溃
                    cwd=Path.cwd(),
                    env=env
                )

                # 实时读取输出并写入日志文件
                for line in process.stdout:
                    clean_line = strip_ansi_codes(line)  # 移除颜色码
                    print(clean_line, end='')  # 输出到控制台
                    f.write(clean_line)  # 写入日志文件
                    f.flush()

                # 等待命令完成
                return_code = process.wait()

                # 写入结束信息
                f.write("=" * 50 + "\n")
                f.write(f"结束时间: {datetime.now()}\n")
                f.write(f"返回码: {return_code}\n")

                if return_code == 0:
                    print(f"\n✓ 命令执行成功，日志已保存到: {log_file}")
                else:
                    print(f"\n✗ 命令执行失败，返回码: {return_code}")
                    print(f"  日志文件: {log_file}")

        except Exception as e:
            print(f"\n✗ 执行命令时发生错误: {e}")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n执行错误: {e}\n")


if __name__ == "__main__":
    # 使用示例
    steps_file = "configs/step.txt"  # 步骤文件路径
    run_commands_from_file(steps_file)
    print("已执行")