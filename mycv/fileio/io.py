from pathlib import Path
from io import BytesIO, StringIO
from typing import Any, Callable, Dict, List, Optional, TextIO, Union
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler
from .file_client import FileClient

FileLikeObject = Union[TextIO, StringIO, BytesIO]

file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler()
}

def load(file: Union[str, Path, FileLikeObject],
         file_format: Optional[str] = None,
         file_client_args: Optional[Dict] = None,
         **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Note:
        In v1.3.16 and later, ``load`` supports loading data from serialized
        files those can be storaged in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet
        >>> load('s3://path/of/your/file')  # file is storaged in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    handler = file_handlers[file_format]
    f: FileLikeObject
    if isinstance(file, str):
        file_client = FileClient.infer_client(file_client_args, file)
        if handler.str_like:
            with StringIO(file_client.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(file_client.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj