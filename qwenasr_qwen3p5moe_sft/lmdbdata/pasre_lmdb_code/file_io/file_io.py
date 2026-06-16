"""
@File    : file_io.py
@Time    : 2024-05-20 15:48:43
@Author  : Fei Jie
@Version : 0.0
@Contact : feijie4@iflytek.com
@License : (C)Copyright XXX
@Desc    : None
"""

# here put the import lib
import os
from abc import abstractmethod
from typing import ByteString, Generic, Iterable, Optional, Tuple, TypeVar

Data = TypeVar("Data")
LmdbKV = Tuple[ByteString, ByteString]


class FileIO(Generic[Data]):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path: str = file_path
        self._is_opened: bool = False

    @abstractmethod
    def __len__(self) -> int:
        """get total number of items in file"""
        raise NotImplementedError

    @property
    def n(self) -> int:
        """get total number of items in file"""
        return len(self)

    @abstractmethod
    def open(self) -> "FileIO":
        """open file"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """close file"""
        raise NotImplementedError

    def __enter__(self) -> None:
        self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        self.close()

    def __del__(self) -> None:
        if self._is_opened:
            self.close()

    def is_open(self) -> bool:
        """whether the file has been opened"""
        return self._is_opened

    def check_opened(self) -> None:
        if not self._is_opened:
            self.open()


class FileReader(FileIO[Data]):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File <{file_path}> not found!")

    @abstractmethod
    def read(self, index: int) -> Data:
        """read the file by index"""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Data:
        return self.read(index)

    def reads(self, indices: Iterable[int]) -> Tuple[Data]:
        return (self.read(i) for i in indices)

    @abstractmethod  # (TODO) rename a better name ?
    def chunk_iter(self, start: int, end: int) -> Iterable[Data]:
        """get a iterator of a contiguious chunk"""
        raise NotImplementedError


class FileWriter(FileIO[Data]):
    @abstractmethod
    def write(self, item: Data) -> None:
        """append a item to file"""
        raise NotImplementedError

    def writes(self, items: Iterable[Data]) -> None:
        """append a list of item to file"""
        for item in items:
            self.write(item)

    @abstractmethod
    def flush(self) -> None:
        """flush the cache"""
        raise NotImplementedError

    def __del__(self) -> None:
        self.flush()
        if self._is_opened:
            self.close()
