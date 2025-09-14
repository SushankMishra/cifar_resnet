import torch

def train_one_epoch(model, device, dataloader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss, correct, count = 0, 0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * xb.size(0)
        correct += out.argmax(dim=1).eq(yb).sum().item()
        count += xb.size(0)
    return total_loss / count, correct / count


def eval_one_epoch(model, device, dataloader, criterion):
    model.eval()
    total_loss, correct, count = 0, 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            correct += out.argmax(dim=1).eq(yb).sum().item()
            count += xb.size(0)
    return total_loss / count, correct / count