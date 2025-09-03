from collections import OrderedDict
from typing import Optional, Union, Iterator, Tuple, Set
from numNet import Tensor
from .. import Parameter

class Module(object):
    training: bool

    def __init__(self) -> None:
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        if module is None:
            self._modules[name] = None
        else:
            self._modules[name] = module

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse = recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        memo = set()
        modules = self.named_modules(prefix = prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            params = module._parameters.items()
            for k, v in params:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def children(self) -> Iterator['Module']:
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def train(self, mode: bool = True) -> 'Module':
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        return self.train(False)

    def __call__(self, *input, **kwargs) -> Tensor:
        out = self.forward(*input, **kwargs)
        return out

    def __setattr__(self, name: str, value):
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self.modules:
            del self._modules[name]
