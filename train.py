import torch
#this methods are borrowed from CS231n assignment-2
#pytorch.ipynb file

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training set")
    else:
        print("Checking accuracy on test set")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            #you can set is equal to "cuda"
            #if you are using gpu
            x = x.to(device='cpu', dtype=torch.float32)
            y = y.to(device='cpu', dtype=torch.long)

            scores = model(x)
            _, pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train(model, optimizer, loader_train, loader_val, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - loader_train : A training dataset
    - losder_val : validation dataset
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device='cpu')  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device='cpu', dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device='cpu', dtype=torch.long)

            scores = model(x)
            loss = torch.nn.functional.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            print_every = 0
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()