# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ConfigMixin base class and utilities."""
import functools
import inspect

from collections import OrderedDict
from typing import Any, Dict, Tuple, Union
from dataclasses import fields

import logging
from typing import List, Optional, Tuple, Union
import torch

logger = logging.getLogger(__name__)


def register_to_config(init):
    r"""
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        # if not isinstance(self, ConfigMixin):
        #     raise RuntimeError(
        #         f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
        #         "not inherit from `ConfigMixin`."
        #     )

        ignore = getattr(self, "ignore_for_config", [])
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default
            for i, (name, p) in enumerate(signature.parameters.items())
            if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )
        new_kwargs = {**config_init_kwargs, **new_kwargs}
        getattr(self, "register_to_config")(**new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `BaseOutput` directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
