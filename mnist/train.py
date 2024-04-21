from torch import nn, optim
import torch
from download import read_mnist_dataset


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layer(x)


def train(model, trainset, epochs=3):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    for e in range(epochs):
        running_loss = 0
        for idx, (images, labels) in enumerate(trainloader):
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()

            if idx and idx % 400 == 0:
                print(
                    f"Epoch {e} - {idx+1} / {len(trainloader)} : {running_loss / idx}"
                )

        print(f"Epoch {e} : {running_loss / len(trainloader)}")

    return model


def score(model, valset):
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    correct_count, all_count = 0, 0
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            ps = ps.to("cpu")
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]

            if true_label == pred_label:
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))


if __name__ == "__main__":
    model = Net()
    trainset, valset = read_mnist_dataset()
    # trained_model = train(model, trainset, epochs=5)
    # results = score(trained_model, valset)
