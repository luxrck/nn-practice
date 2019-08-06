import os
import re
import sys
import functools
from collections import defaultdict, deque

import torch
from torch import nn, optim
import torch.nn.functional as F

from tqdm import tqdm



class Checkpoint(object):
    def __init__(self, app):
        self.app = app
        self.fastforwarded = False
        self.checkpoint_root = os.path.join(self.app_root, "checkpoint")

    def fastforward(self, index=-1):
        @self.app.on("train_started")
        def forward(e):
            self._fastforward(index=index)
            e.trainer.current_iter = self.current_iter
        return self
    def _fastforward(self, index=-1):
        model_name = self.model.__class__.__name__
        checkpoint = self.checkpoint_root
        if not os.path.exists(checkpoint):
            os.mkdir(checkpoint)
        state = None
        key = lambda x: re.search(f"(?<={model_name}\.)(\d+)(?=\.pt)", x)
        fastforward = list(filter(key, list(os.walk(checkpoint))[0][2]))
        if fastforward:
            fastforward = sorted(fastforward, key=lambda x: int(key(x).group(0)))
            model_location = os.path.join(checkpoint, fastforward[index])
            state = torch.load(model_location, map_location=self.device)
        if state:
            self.current_iter = state["start"]
            self.model.load_state_dict(state["model"])
            # By default all the modules are initialized to train mode (self.training = True).
            # Also be aware that some layers have different behavior during train/and evaluation
            # (like BatchNorm, Dropout) so setting it matters.
            self.model.train()
            self.optimizer.load_state_dict(state["optim"])
        return self

    def save_every(self, iters=1000):
        @self.on("iter_completed")
        def save(e):
            current_iter = e.current_iter
            if current_iter % iters == 0:
                torch.save({"start": current_iter + 1, "model": self.model.state_dict(), "optim": self.optimizer.state_dict()},
                           os.path.join(self.checkpoint_root, f"{self.model.__class__.__name__}.{current_iter}.pt"))
        return self

    # Called when the default attribute access fails with an AttributeError (either __getattribute__() raises an
    # AttributeError because name is not an instance attribute or an attribute in the class tree for self; or __get__()
    # of a name property raises AttributeError). This method should either return the (computed) attribute value or raise
    # an AttributeError exception.
    # https://docs.python.org/3/reference/datamodel.html#object.__getattr__
    def __getattr__(self, k):
        if self.fastforwarded:
            raise AttributeError("Can not call `self.to` or `self.half` after `Checkpoint::fastforwarded`.")
        try:
            return self.app.__getattribute__(k)
        except AttributeError:
            return self.app.__getattr__(k)



class Trainer(object):
    r'''
    Events:
        train_started:
        iter_started:
        iter_completed:
        train_completed:
    '''

    class Event(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __setattr__(self, k, v):
            self.__dict__[k] = v

    def __init__(self, app):
        self.app = app
        self.event_map = defaultdict(set)
        self.current_iter = 1

    def exec_handles(self, on_event, e):
        deque(map(lambda h: h(e), self.event_map[on_event]))

    def on(self, event):
        def event_wrapper(event_handler):
            def inner_event_wrapper(*args, **kwargs):
                return event_handler(*args, **kwargs)
            self.event_map[event].add(inner_event_wrapper)
            return inner_event_wrapper
        return event_wrapper

    def set_optimizer(self, op, *args, **kwargs):
        self.optimizer_builder = functools.partial(op, self.model.parameters(), *args, **kwargs)
        return self

    def run(self, data_iter, max_iters=1000, train=True):
        self.optimizer = self.optimizer_builder()

        self.exec_handles("train_started",
                           Trainer.Event(name="train_started", trainer=self))

        current_iter = self.current_iter

        if train == False:
            return self.exec_handles("train_completed",
                                      Trainer.Event(name="train_completed", trainer=self))

        while current_iter < max_iters + 1:
            iterator = tqdm(enumerate(data_iter))
            for i,batch in iterator:
                self.exec_handles("iter_started",
                                   Trainer.Event(name="iter_started", batch=batch, model=self.model, criterion=self.criterion, optimizer=self.optimizer))
                self.exec_handles("iter_completed",
                                   Trainer.Event(name="iter_completed", current_iter=current_iter))
                current_iter += 1
                if current_iter >= max_iters + 1:
                    break

        return self.exec_handles("train_completed",
                                  Trainer.Event(name="train_completed", trainer=self))

    def __getattr__(self, k):
        return self.app.__getattribute__(k)



class App(object):
    def __init__(self, name="", root=".", **kwargs):
        self.name = name or os.path.basename(sys.modules[__name__].__file__)
        self.config(**kwargs)
        self.app_root = root

    def config(self, **kwargs):
        default = {
            "device": "cpu",
            "model": None,
            "criterion": None,
            }
        default.update(kwargs)
        self.c = default
        for key,val in default.items():
            self.__setattr__(key, val)
        return self

    def to(self, device):
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        return self

    def half(self):
        if torch.cuda.is_available():
            self.model, self.criterion = self.model.half(), self.criterion.half()
        return self

    def build_optimizer(self, op, *args, **kwargs):
        self.optimizer = op(self.model.parameters(), *args, **kwargs)
        return self