# CropForegroundFixedD = CropForegroundFixedDict = CropForegroundFixedd
import torch
import numpy as np

from monai.transforms import (
    BorderPad,
    Crop,
    Cropd,
    Pad
)

from itertools import chain
from monai.config import KeysCollection
from typing import Sequence, Union, Optional, Mapping, Hashable, Callable, Dict
from monai.transforms.transform import LazyTransform
from monai.transforms.utils import (
    generate_spatial_bounding_box,
    is_positive,
)
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.utils import PytorchPadMode,TraceKeys, ensure_tuple, ensure_tuple_rep, convert_data_type
from monai.config import IndexSelection, SequenceStr

#Augmentation logic
class CropForegroundFixed(Crop):
    def __init__(
        self,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        allow_smaller: bool = True,
        return_coords: bool = False,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: str = PytorchPadMode.CONSTANT,
        lazy: bool = False,
        **pad_kwargs,
    ) -> None:

        LazyTransform.__init__(self, lazy)
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.spatial_size = spatial_size
        self.allow_smaller = allow_smaller
        self.return_coords = return_coords
        self.k_divisible = k_divisible
        self.padder = Pad(mode=mode, lazy=lazy, **pad_kwargs)
    
    @Crop.lazy.setter  # type: ignore
    def lazy(self, _val: bool):
        self._lazy = _val
        self.padder.lazy = _val

    @property
    def requires_current_data(self):
        return False

    def compute_bounding_box(self, img: torch.Tensor):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(
            img, self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        # make the spatial size divisible by `k`
        # spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        center_ = np.floor_divide(box_start_ + box_end_, 2)
        box_start_ = center_ - np.floor_divide(np.asarray(self.spatial_size), 2)
        box_end_ = box_start_ + self.spatial_size

        return box_start_, box_end_

    def crop_pad(
        self, img: torch.Tensor, box_start: np.ndarray, box_end: np.ndarray, mode: Optional[str] = None, lazy: bool = False, **pad_kwargs
    ):
        """
        Crop and pad based on the bounding box.

        """        
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = super().__call__(img=img, slices=slices)
        pad_to_start = np.maximum(-box_start, 0)
        if len(img.shape)==4:
            pad_to_end = np.maximum(box_end - np.asarray(img.shape[1:]), 0)
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            pad_width = BorderPad(spatial_border=pad).compute_pad_width(cropped.shape[1:])
        else:
            pad_to_end = np.maximum(box_end - np.asarray(img.shape[2:]), 0)
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            pad_width = BorderPad(spatial_border=pad).compute_pad_width(cropped.shape[2:])
        ret = self.padder.__call__(img=cropped, to_pad=pad_width, mode=mode, **pad_kwargs)
        # combine the traced cropping and padding into one transformation
        # by taking the padded info and placing it in a key inside the crop info.
        if get_track_meta():
            ret_: MetaTensor = ret  # type: ignore
            app_op = ret_.applied_operations.pop(-1)
            ret_.applied_operations[-1][TraceKeys.EXTRA_INFO]["pad_info"] = app_op
        return ret

    def __call__(self, img: torch.Tensor, mode: Optional[str] = None, lazy: Optional[bool] = None, **pad_kwargs):  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = self.compute_bounding_box(img)
        lazy_ = self.lazy if lazy is None else lazy
        cropped = self.crop_pad(img, box_start, box_end, mode, lazy=lazy_, **pad_kwargs)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.get_most_recent_transform(img)
        # we moved the padding info in the forward, so put it back for the inverse
        pad_info = transform[TraceKeys.EXTRA_INFO].pop("pad_info")
        img.applied_operations.append(pad_info)
        # first inverse the padder
        inv = self.padder.inverse(img)
        # and then inverse the cropper (self)
        return super().inverse(inv)

#Augmentation dictionary wrapper
class CropForegroundFixedd(Cropd):

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        allow_smaller: bool = True,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: SequenceStr = PytorchPadMode.CONSTANT,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **pad_kwargs,
    ) -> None:

        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        cropper = CropForegroundFixed(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            spatial_size=spatial_size,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            lazy=lazy,
            **pad_kwargs,
        )
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
    
    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value

    @property
    def requires_current_data(self):
        return True
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: Optional[bool] = None) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.cropper: CropForegroundFixed
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        lazy_ = self.lazy if lazy is None else lazy
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_)
        return d

CropForegroundFixedD = CropForegroundFixedDict = CropForegroundFixedd