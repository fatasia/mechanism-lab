"""公开数据下载与远端ZIP只读访问；不执行数据中的脚本，不展开不可信路径。"""

import hashlib
import io
import json
import subprocess
import shutil
from collections import OrderedDict
from pathlib import Path


def curl_bytes(url, byte_range=None):
    if not url.startswith("https://"):
        raise ValueError("只接受HTTPS公开地址")
    command = [
        shutil.which("curl.exe") or "curl",
        "-fLsS",
        "--retry",
        "3",
        "--retry-all-errors",
        "--retry-max-time",
        "240",
        "--max-time",
        "60",
    ]
    if byte_range is not None:
        command += ["--range", f"{byte_range[0]}-{byte_range[1]}"]
    result = subprocess.run([*command, url], capture_output=True, check=True)
    if byte_range and len(result.stdout) != byte_range[1] - byte_range[0] + 1:
        raise ValueError("服务器未按请求返回Range，拒绝把完整归档当作分块")
    return result.stdout


def read_json(url):
    return json.loads(curl_bytes(url))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RemoteZipFile(io.RawIOBase):
    """基于发布方已知文件长度的HTTP Range文件；缓存最多16MiB。"""

    def __init__(self, url, size, cache_dir):
        self.url, self.size, self.position = url, size, 0
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.blocks = OrderedDict()
        self.block_size = 1 << 20
        self.downloaded_bytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        if value < 0:
            raise ValueError("负偏移")
        self.position = value
        return value

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            start = self.position // self.block_size * self.block_size
            if start not in self.blocks:
                path = self.cache_dir / f"{start}.bin"
                expected = min(self.block_size, self.size - start)
                if not path.is_file() or path.stat().st_size != expected:
                    data = curl_bytes(self.url, (start, start + expected - 1))
                    temporary = path.with_suffix(".partial")
                    temporary.write_bytes(data)
                    temporary.replace(path)
                    self.downloaded_bytes += len(data)
                self.blocks[start] = path.read_bytes()
                if len(self.blocks) > 16:
                    self.blocks.popitem(last=False)
            data = self.blocks[start]
            count = min(end - self.position, len(data) - (self.position - start))
            chunks.append(data[self.position - start : self.position - start + count])
            self.position += count
        return b"".join(chunks)
