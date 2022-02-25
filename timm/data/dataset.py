""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import torch.utils.data as data
import os
import torch
import logging
import six

import numpy as np
from PIL import Image
try:
    import jpeg4py as jpeg
except:
    pass
try:
    import cv2
except:
    pass
try:
    import lmdb
    import msgpack
    import pyarrow as pa
except:
    pass

from .parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
            img_load='PIL',
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.img_load_func = {
            'numpy': lambda img: Image.fromarray(np.load(img)),
            'PIL': lambda img: Image.open(img).convert('RGB'),
            'jpeg4py': lambda img: Image.fromarray(jpeg.JPEG(img).decode()),
            'cv2': lambda img: Image.fromarray(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
        }[img_load]

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else self.img_load_func(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training,
                batch_size=batch_size, repeats=repeats, download=download)
        else:
            self.parser = parser
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)

class ImageFolderLMDB(data.Dataset):
    def __init__(
            self, 
            root, 
            is_training=False,
            transform=None, 
            target_transform=None,
            **kwargs
    ):
        if is_training:
            self.db_path = os.path.join(root, 'train.lmdb')
        else:
            self.db_path = os.path.join(root, 'val.lmdb')

        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'