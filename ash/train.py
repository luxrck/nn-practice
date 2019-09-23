import os
import re
import sys
import functools
import inspect
from collections import defaultdict, deque

import torch
from torch import nn, optim
import torch.nn.functional as F

from tqdm import tqdm



class Checkpoint(object):
    #def __new__(cls, app):
    #    self = super().__new__(cls)
    #    self.__init__(app)
    #    app.register("checkpoint", self)
    #    return app

    __name__ = "checkpoint"

    def __init__(self, root="checkpoint"):
        self.checkpoint_root = root

    def bind(self, app):
        self.app = app
        self.fastforwarded = False
        self.checkpoint_root = os.path.join(app.app_root, self.checkpoint_root)

    def set_checkpoint_root(self, root):
        self.checkpoint_root = os.path.join(self.app.app_root, root)
        return self

    def fastforward(self, index=-1):
        @self.app.on("train_started")
        def forward(e):
            self._fastforward(index=index)
        return self
    def _fastforward(self, index=-1):
        app = self.app
        checkpoint = self.checkpoint_root
        model_name = app.model.__class__.__name__
        if not os.path.exists(checkpoint):
            os.mkdir(checkpoint)
        state = None
        key = lambda x: re.search(f"(?<={model_name}\.)(\d+)(?=\.pt)", x)
        fastforward = list(filter(key, list(os.walk(checkpoint))[0][2]))
        if fastforward:
            fastforward = sorted(fastforward, key=lambda x: int(key(x).group(0)))
            model_location = os.path.join(checkpoint, fastforward[index])
            state = torch.load(model_location, map_location=app.device)
        if state:
            app.current_iter = state["start"]
            app.model.load_state_dict(state["model"])
            # By default all the modules are initialized to train mode (self.training = True).
            # Also be aware that some layers have different behavior during train/and evaluation
            # (like BatchNorm, Dropout) so setting it matters.
            app.model.train()
            app.optimizer.load_state_dict(state["optim"])
        return self

    def save_every(self, iters=1000):
        @self.app.on("iter_completed")
        def save(e):
            current_iter = e.current_iter
            if current_iter % iters == 0:
                torch.save({"start": current_iter + 1, "model": e.model.state_dict(), "optim": e.optimizer.state_dict()},
                           os.path.join(self.checkpoint_root, f"{e.model.__class__.__name__}.{current_iter}.pt"))
        return self



class Trainer(object):
    r'''
    Events:
        train_started:
        iter_started / train:
        iter_completed:
        train_completed:

        evaluate:
    '''

    class Event(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __setattr__(self, k, v):
            self.__dict__[k] = v

    def __init__(self, app):
        self.app = app
        self.event_map = defaultdict(set)
        self.event_map["train"] = self.event_map["iter_started"]

        self.extension_map = {}
        self.current_iter = 1

    def extend(self, ext):
        ext.bind(self)
        self.extension_map[ext.__name__] = ext
        return self

    def exec_handles(self, on_event, e):
        return list(map(lambda h: h(e), self.event_map[on_event]))

    def on(self, event, handler=None):
        def event_wrapper(handler):
            def inner_event_wrapper(*args, **kwargs):
                return handler(*args, **kwargs)
            self.event_map[event].add(inner_event_wrapper)
            return inner_event_wrapper
        if handler is not None:
            return event_wrapper(handler)
        return event_wrapper

    def set_optimizer(self, op, *args, **kwargs):
        if issubclass(op, torch.optim.Optimizer):
            op = functools.partial(op, self.model.parameters())
        else:
            op = functools.partial(op, self)
        self.optimizer_builder = functools.partial(op, *args, **kwargs)
        return self

    def eval(self, data):
        self.model.eval()

        oy_p, oy_g = [], []

        with torch.no_grad():
            for _,batch in tqdm(enumerate(data)):
                # TODO: Not a good implementation...
                results = self.exec_handles("evaluate",
                                                 Trainer.Event(name="evaluate", trainer=self, batch=batch, model=self.model, criterion=self.criterion, optimizer=self.optimizer))[0]
                if results is not None and len(results) == 2:
                    y_predicted, targets = results
                    oy_p.append(y_predicted)
                    oy_g.append(targets)
        oy_p = torch.cat(oy_p, dim=0)
        oy_g = torch.cat(oy_g, dim=0)
        return oy_p, oy_g

    def run(self, data, max_iters=1000, train=True):
        self.model.train()
        self.optimizer = self.optimizer_builder()

        meta = Trainer.Event(name="meta")
        progress = tqdm(total=max_iters, miniters=0)
        event = Trainer.Event(name="e", a=meta, progress=progress, trainer=self, model=self.model, criterion=self.criterion, optimizer=self.optimizer)

        self.exec_handles("train_started", event)

        current_iter = self.current_iter
        event.progress.n = current_iter
        event.progress.last_print_n = current_iter

        if train == False:
            self.exec_handles("train_completed", event)
            return self

        while current_iter < max_iters + 1:
            iterator = enumerate(data)
            for i,batch in iterator:
                event.current_iter = current_iter
                event.batch = batch
                self.exec_handles("iter_started", event)
                self.exec_handles("iter_completed", event)
                current_iter += 1
                if current_iter >= max_iters + 1:
                    break
                event.progress.update(1)
        self.exec_handles("train_completed", event)
        return self

    # Called when the default attribute access fails with an AttributeError (either __getattribute__() raises an
    # AttributeError because name is not an instance attribute or an attribute in the class tree for self; or __get__()
    # of a name property raises AttributeError). This method should either return the (computed) attribute value or raise
    # an AttributeError exception.
    # https://docs.python.org/3/reference/datamodel.html#object.__getattr__
    # Trainer包装App, 当Train没有属性k时, 可以从App中查找, 但最后要返回Trainer的Instance.
    def __getattr__(self, k):
        class Self(object):
            def __init__(self, prev_self, chained):
                self._self = prev_self
                self.chained = chained
            def __call__(self, *args, **kwargs):
                self.chained(*args, **kwargs)
                return self._self
        v = None
        for _,ext in self.extension_map.items():
            try:
                v = ext.__getattribute__(k)
                break
            except AttributeError:
                pass
        if not v:
            v = self.app.__getattribute__(k)
        # import pdb; pdb.set_trace()
        if inspect.ismethod(v):
            return Self(self, v)
        return v



class App(object):
    def __init__(self, name="", root=".", **kwargs):
        self.app_name = name or os.path.basename(sys.modules[__name__].__file__)
        self.app_root = root
        self.config(**kwargs)

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
