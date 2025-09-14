from torch_lr_finder import LRFinder

def run_lr_finder(model, optimizer, criterion, dataloader, device, end_lr=1, num_iter=100):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(dataloader, end_lr=end_lr, num_iter=num_iter)
    lr_finder.plot()  # matplotlib figure
    lr_finder.reset()