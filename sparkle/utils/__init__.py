import torch
from tqdm import tqdm
import os
import re
import sys

from torchtext.data.iterator import Iterator

# We will exec `fn` immediately instead of return a wrapper function
def train(model, criterion, optimizer, dataloader, epochs, device=None, half=False, checkpoint=None, validation=None, immediate=False, verbose=False, fn=None):
    start = 0
    model_name = model.__class__.__name__

    # %%
    # 让model这么早就映射到`device`的原因是:
    # 如果我们打开`checkpoint`开关, 对于一些优化器, 我们需要加载它们之前被保存的状态,
    # 这就带来了一个问题: Optimizer::load_state_dict的实现(torch v1.1.0)是自动将state
    # 映射到`model.parameters()`所在的`device`上面, 并且将state的dtype设置成和模型参数一样的dtype.
    # 如果我们不先把model映射好, load_state之后如果我们要改变model的device的话, 会造成optimizer
    # 的参数位置和model的不在同一设备上, 即出错.
    # 所以我们必须要一开始就先把model给映射好. 并且将模型的参数类型也要设置好(比如是否用half)
    # %%
    # 那么问题来了, model.load_state_dict会不会改变model的device呢?
    # 答: 不会. model.load_state_dict加载参数最终调用的是Tensor::copy_, 这并不会改变Tensor的device.
    # 所以我们可以放心地在一开始就映射好model
    model = model.to(device)
    criterion = criterion.to(device)
    if half:
        model, criterion = model.half(), criterion.half()

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint:
        if checkpoint == True:
            checkpoint = "./checkpoint"
        if not os.path.exists(checkpoint):
            os.mkdir(checkpoint)
        state = None
        key = lambda x: re.search(f"(?<={model_name}\.)(\d+)(?=\.torch)", x)
        fastforward = list(filter(key, list(os.walk(checkpoint))[0][2]))
        if fastforward:
            fastforward = max(fastforward, key=lambda x: int(key(x).group()))
            fastforward = os.path.join(checkpoint, fastforward)
            state = torch.load(fastforward, map_location=device)
        if state:
            start = state["start"]
            model.load_state_dict(state["model"])
            # By default all the modules are initialized to train mode (self.training = True).
            # Also be aware that some layers have different behavior during train/and evaluation
            # (like BatchNorm, Dropout) so setting it matters.
            model.train()
            optimizer.load_state_dict(state["optim"])

    def to_device(inputs, labels, device):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = torch.cat([i.view(1, *i.size()) for i in inputs], dim=0).to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        if half and inputs.is_floating_point():
            inputs = inputs.half()
        return inputs, labels

    def _validate(fn, dataloader):
        correct, total = 0, 0
        with torch.no_grad():
            for i,(inputs,labels) in tqdm(enumerate(dataloader)):
                inputs, labels = to_device(inputs, labels, device)
                y_predict, _ = fn(model, criterion, inputs, labels)
                _, l_predict = torch.max(y_predict, dim=1)
                correct_cnt = (l_predict == labels).sum()
                total_cnt = labels.size(0)
                correct += correct_cnt.item()
                total += total_cnt
        return correct, total

    def _train(fn):
        for epoch in range(start, epochs):
            total_loss = 0
            # import pdb; pdb.set_trace()
            for i,(data) in tqdm(enumerate(dataloader)):
                model.zero_grad()
                # inputs, labels = (data.src, data.trg) if isinstance(dataloader, Iterator) else data
                # inputs, labels = to_device(inputs, labels, device)

                loss = fn(model, criterion, data)

                loss.backward()
                optimizer.step()
                #import pdb; pdb.set_trace()

                total_loss += loss.item() # / inputs.size(0)
                if verbose:
                    if i % 100 == 0:
                        print(f"epoch: {epoch}, iter: {i}, loss: {loss}, total_loss: {total_loss}")

            if checkpoint:
                torch.save({"start": epoch + 1, "model": model.state_dict(), "optim": optimizer.state_dict()},
                            os.path.join(checkpoint, f"{model.__class__.__name__}.{epoch}.torch"))

            correct, total = 0, 0
            printed_str = f"epoch: {epoch}, loss: {total_loss}"
            if validation:
                correct, total = _validate(fn, validation)
                printed_str += " " + f"validate: {total}, {correct/total}"
            print(printed_str)
        return model

    def _train_wrapper(fn):
        if immediate:
            return _train(fn)
        def _train_deco():
            return _train(fn)
        return _train_deco

    if fn: return _train(fn)
    return _train_wrapper


def test(model, criterion, optimizer, testloader, test_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    def _test(test_fn):
        with torch.no_grad():
            count = 0
            total = 0
            for i, (inputs, labels) in tqdm(enumerate(testloader)):
                inputs, labels = inputs.to(device), labels.to(device)
                correct_cnt = test_fn(model, criterion, inputs, labels)
                total_cnt = len(labels)
                count += correct_cnt
                total += total_cnt
            print("Acc:", int(count), int(total), int(count)/int(total))
    def _deco(test_fn):
        def __deco():
            return _test(test_fn)
        return __deco
    if test_fn: return _test(test_fn)
    return _deco
