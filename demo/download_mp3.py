import requests
import os
from concurrent.futures import ThreadPoolExecutor


def download_mp3(base_url, prefix, start_num, end_num, output_dir="D:\\DolphinProjects\\Dolphin-WC.AudiometryService\\PureToneSound"):
    """
    下载指定范围内的MP3文件

    参数:
    base_url: 基础URL (例如: "https://zj.body120.xyz/amAudio_new_3/")
    prefix: 文件前缀 (例如: "L" 或 "R")
    start_num: 起始编号 (例如: 5000)
    end_num: 结束编号 (例如: 5010)
    output_dir: 保存文件的目录 (默认为"downloads")
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    def download_single(num):
        filename = f"{prefix}{num:05d}.mp3"  # 格式化为5位数，如L5000.mp3
        url = f"{base_url}{filename}"
        save_path = os.path.join(output_dir, filename)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 检查请求是否成功

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"成功下载: {filename}")
            return True
        except Exception as e:
            print(f"下载失败 {filename}: {str(e)}")
            return False

    # 使用线程池并发下载
    with ThreadPoolExecutor(max_workers=5) as executor:
        nums = range(start_num, end_num + 1)
        results = list(executor.map(download_single, nums))

    success_count = sum(results)
    print(f"\n下载完成! 成功: {success_count}, 失败: {len(results) - success_count}")


if __name__ == "__main__":
    # 基础URL
    base_url = "https://zj.body120.xyz/amAudio_new_3/"

    # 用户输入参数
    prefix = input("请输入文件前缀(例如 L 或 R): ").strip().upper()
    start_num = int(input("请输入起始编号(例如 5000): "))
    end_num = int(input("请输入结束编号(例如 5010): "))

    # 调用下载函数
    download_mp3(base_url, prefix, start_num, end_num)