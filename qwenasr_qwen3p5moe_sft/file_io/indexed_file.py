"""
@File    : indexed_file.py
@Time    : 2024-05-20 15:52:26
@Author  : Fei Jie
@Version : 0.0
@Contact : feijie4@iflytek.com
@License : (C)Copyright XXX
@Desc    : None
"""

# here put the import lib
import os
import struct
from typing import Dict, Iterable, List, Union

import numpy as np

from .file_io import FileIO, FileReader, FileWriter

__all__ = ["LengthsFileReader", "LengthsFileWriter"]


class LengthsIndexedFileIO(FileIO):
    _pack_fmt = "<I"  # uint32

    def __init__(self, file_path, mode) -> None:
        super(LengthsIndexedFileIO, self).__init__(file_path)
        self._file_path: str = file_path
        self._mode: str = mode
        self._fp = None

    def open(self) -> "LengthsIndexedFileIO":
        if self._is_opened:
            return
        self._fp = open(self.file_path, self._mode)
        self._is_opened = True
        return self

    def close(self) -> None:
        if self._is_opened:
            self._fp.close()
            self._is_opened = False

    def __len__(self) -> int:
        file_size = os.path.getsize(self._file_path)
        return file_size // self.itemsize()

    def _make_fmt(self, size) -> str:
        fmt = self._pack_fmt
        return f"{fmt[0]}{size}{fmt[1]}"

    @classmethod
    def set_fmt(cls, fmt) -> None:
        cls._pack_fmt = fmt

    @classmethod
    def get_fmt(cls) -> str:
        return cls._pack_fmt

    @classmethod
    def itemsize(cls) -> int:
        return struct.calcsize(cls._pack_fmt)


class LengthsFileReader(LengthsIndexedFileIO, FileReader):
    def __init__(self, file_path: str) -> None:
        super(LengthsFileReader, self).__init__(file_path, "rb")
        self._dtype = np.dtype(self._pack_fmt)
        self._bin_mmap: np.memmap = None
        self._bin_mview: memoryview = None

    def open(self) -> "LengthsFileReader":
        super(LengthsFileReader, self).open()
        self._bin_mmap = np.memmap(self._fp, mode="r", order="C")
        self._bin_mview = memoryview(self._bin_mmap)
        self._buffer = np.frombuffer(self._bin_mview, dtype=self._dtype).copy()
        self._bin_mmap._mmap.close()
        del self._bin_mmap
        del self._bin_mview
        return self

    def close(self) -> None:
        super(LengthsFileReader, self).close()
        if self._is_opened:
            del self._buffer
            self._is_opened = False

    def __getstate__(self) -> Dict:
        state = {
            "file_path": self.file_path,
            "is_opened": self._is_opened,
        }
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__init__(state["file_path"])
        is_opened = state["is_opened"]
        if is_opened:
            self.open()

    def read(self, index: int) -> int:
        self.check_opened()
        return self._buffer[index].item()

    def __iter__(self) -> Iterable[int]:
        for i in range(self.n):
            yield self.read(i)

    def __getitem__(self, index) -> Union[int, List[int]]:
        if not isinstance(index, slice):
            return self.read(int(index))
        return [self.read(i) for i in range(index.start, index.stop, index.step)]


class LengthsFileWriter(LengthsIndexedFileIO, FileWriter):
    """We use `struct` instead of `numpy`, because the former is faster than the latter"""

    buffering = 1024 * 1024  # 1MB

    def __init__(self, file_path: str) -> None:
        super(LengthsFileWriter, self).__init__(file_path, "wb")

    def write(self, item: int) -> None:
        assert isinstance(item, int) and self._is_opened
        bytestring = struct.pack(self._pack_fmt, item)
        self._fp.write(bytestring)
        if self._fp.tell() % self.buffering == 0:
            self._fp.flush()

    def writes(self, items: Iterable[int]) -> None:
        self.check_opened()
        cache_size = self.buffering / self.itemsize()
        cache = list()
        for i in items:
            cache.append(i)
            assert isinstance(i, int)
            if len(cache) == cache_size:
                self._fp.write(struct.pack(self._make_fmt(len(cache)), *cache))
                self.flush()
                cache.clear()
        if len(cache) != 0:
            self._fp.write(struct.pack(self._make_fmt(len(cache)), *cache))
            self.flush()
            cache.clear()

    def flush(self) -> None:
        if not self._is_opened:
            return
        self._fp.flush()

    def __len__(self) -> int:
        if not self._is_opened:
            return super().__len__()
        return self._fp.tell() // self.itemsize()
