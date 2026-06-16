"""
@File    : lmdb_file.py
@Time    : 2024-05-20 15:51:46
@Author  : Fei Jie
@Version : 0.0
@Contact : feijie4@iflytek.com
@License : (C)Copyright XXX
@Desc    : None
"""

# here put the import lib
from typing import ByteString, Dict, Iterable, Tuple, Union

import lmdb

from .file_io import FileIO, FileReader, FileWriter

__all__ = ["LmdbReader", "LmdbWriter"]


LmdbKV = Tuple[ByteString, ByteString]


class LmdbFileIO(FileIO):
    max_volume = 10**12
    default_key_fmt: str = "{}"
    increament_key_fmt: str = "{:011d}"

    def __init__(
        self, file_path: str, subdir: bool = False, increament_key: bool = False
    ) -> None:
        super(LmdbFileIO, self).__init__(file_path)
        self.subdir: bool = subdir
        self.increament_key: bool = increament_key
        self._key_fmt: str = self.default_key_fmt
        if increament_key:
            self._key_fmt = self.increament_key_fmt

    def open(self) -> "LmdbFileIO":
        """open lmdb file and create a readonly enviroment"""
        return self

    def close(self) -> None:
        """close lmdb environment"""
        pass

    def __getstate__(self) -> Dict:
        state = {
            "file_path": self.file_path,
            "subdir": self.subdir,
            "increament_key": self.increament_key,
        }
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__init__(state["file_path"], state["subdir"], state["increament_key"])

    def __len__(self) -> int:
        """total number of items in lmdb file"""
        env = lmdb.open(self.file_path, subdir=self.subdir, readonly=True, lock=False, readahead=False)
        size = int(env.stat()["entries"])
        return size

    def set_fmt(self, key_fmt: str):
        self._key_fmt = key_fmt


class LmdbReader(LmdbFileIO, FileReader):
    def read(self, index: int, return_key: bool = True) -> Union[ByteString, LmdbKV]:
        """get lmdb item by index, the index should be an interger."""
        env = lmdb.open(self.file_path, subdir=self.subdir, readonly=True, lock=False, readahead=False)
        key = self._key_fmt.format(index).encode()
        with env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"subdir={self.file_path} Key={key} Index={index} not found!")
        if not return_key:
            return value
        return key, value

    def read_chunk(self, start:int, end:int, return_key: bool = True) -> Union[ByteString, LmdbKV]:
        env = lmdb.open(self.file_path, subdir=self.subdir, readonly=True, lock=False, readahead=False)
        assert (
            self.increament_key
        ), f"the encoded keys must are self increasing and ordered in lmdb"
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            start_key = self._key_fmt.format(start).encode()
            if not cursor.set_range(start_key):
                raise KeyError(f"set_range key={start_key} fail !")

            cnt = start
            for key, value in cursor:
                if not return_key:
                    yield value
                else:
                    yield key, value
                cnt += 1
                if cnt >= end:
                    break


class LmdbWriter(LmdbFileIO, FileWriter):
    buffering: int = 1000

    def __init__(
        self,
        file_path: str,
        subdir: bool = False,
        increament_key: bool = False,
        map_size: int = 10485760,
    ):
        super(LmdbWriter, self).__init__(file_path, subdir, increament_key)
        self._map_size = map_size
        self._count = 0
        self._cache = list()

    def open(self) -> "LmdbWriter":
        """open lmdb file and create a readonly enviroment"""
        if self._is_opened or self._env is not None:
            return
        self._env = lmdb.Environment(
            self.file_path, subdir=self.subdir, map_size=self._map_size, lock=True
        )
        self._is_opened = True
        self._count = int(self._env.stat()["entries"])
        return self

    def close(self) -> None:
        self.flush()
        super(LmdbWriter, self).close()

    @property
    def count(self) -> int:
        assert self._is_opened
        return self._count

    def write(self, item: bytes) -> None:
        self.check_opened()
        if self._count + len(self._cache) >= self.max_volume:
            self.flush()
            raise OverflowError(f"Max volume({self.max_volume}) exceeded!")

        assert isinstance(item, bytes)
        self._cache.append(item)
        if len(self._cache) >= self.buffering:
            self.flush()

    def writes(self, items: Iterable[bytes]) -> None:
        for i in items:
            self.write(i)

    def flush(self) -> None:
        if not self._is_opened or len(self._cache) <= 0:
            return
        with self.env.begin(write=True) as txn:
            while len(self._cache) > 0:
                item = self._cache.pop(0)
                key = self._key_fmt.format(self._count)
                txn.put(key.encode(), item)
                self._count += 1

    def __len__(self) -> int:
        return self._count
