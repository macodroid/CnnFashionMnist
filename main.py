import torch
from matplotlib import pyplot as plt
import numpy as np
import gzip
import argparse

from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from torchvision import transforms


class MnistFashionDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.transform = transform
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # second block
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            # third block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            # forth block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            # 5 block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

        # Initialize weights
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        # Apply initialization
        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class Dojo:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
    ):
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.device = device

    def train(self):
        train_losses = []
        val_losses = []
        correct_train = 0
        total_train = 0
        self.model.train()
        for i, batch in enumerate(self.train_dl):
            x = batch[0].type(torch.FloatTensor)
            x = x.to(self.device)
            y = batch[1].type(torch.LongTensor)
            y = y.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(x)
            loss = self.loss_fn(out, y)
            acc = torch.sum(torch.argmax(out, dim=-1) == y)
            correct_train += acc.item()
            total_train += len(batch[1])
            loss.backward()
            train_losses.append(loss.item())
            self.optimizer.step()
            # if i % 100 == 0:
            #     print("Training loss at step {}: {}".format(i, loss.item()))
        self.model.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            for i, batch in enumerate(self.val_dl):
                x = batch[0].type(torch.FloatTensor)
                x = x.to(self.device)
                y = batch[1].type(torch.LongTensor)
                y = y.to(self.device)

                out = self.model(x)
                loss = self.loss_fn(out, y)
                acc = torch.sum(torch.argmax(out, dim=-1) == y)
                correct_val += acc.item()
                total_val += len(batch[1])
                val_losses.append(loss.item())

        val_acc = correct_val / total_val
        train_acc = correct_train / total_train
        return np.mean(train_losses), np.mean(val_losses), train_acc, val_acc

    def test(self):
        num_correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                x = batch[0].type(torch.FloatTensor)
                x = x.to(self.device)
                y = batch[1].type(torch.LongTensor)
                y = y.to(self.device)

                y_hat = self.model(x)
                acc = torch.sum(torch.argmax(y_hat, dim=-1) == y)
                num_correct += acc.item()
                total += len(batch[1])

        return float(num_correct) / float(total)


def load_mnist(kind):
    labels_path = "./{}-labels-idx1-ubyte.gz".format(kind)
    images_path = "./{}-images-idx3-ubyte.gz".format(kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )
    return images.reshape(len(images), 28, 28), labels


def create_plot(
    epoch_train_losses,
    epoch_val_losses,
    epoch_train_accs,
    epoch_val_acc,
    name,
    e,
    save_plot=True,
    display_plot=False,
):
    figure, axis = plt.subplots(2, 1)
    axis[0].plot(epoch_train_losses, c="r")
    axis[0].plot(epoch_val_losses, c="b")
    axis[0].legend(["Train_loss", "Val_loss"])
    axis[0].set_title("Train vs. Validation loss")
    axis[0].set(xlabel="Epoch", ylabel="Loss")
    axis[1].plot(epoch_train_accs)
    axis[1].plot(epoch_val_acc)
    axis[1].legend(["Train_acc", "Val_acc"])
    axis[1].set_title("Accuracy")
    axis[1].set(xlabel="Epoch", ylabel="Accuracy")
    plt.tight_layout()
    if save_plot:
        plt.savefig(f"CnnMnist-{e}-{name}.png")
    if display_plot:
        plt.show()


X_train, y_train = load_mnist("train")
X_test, y_test = load_mnist("test")
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
img_rows = 28
img_cols = 28
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)

transformation = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    ]
)

train_dataset = MnistFashionDataset(X_train, y_train, transformation)
X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)
test_dataset = TensorDataset(X_test, y_test)

train_set_size = int(len(train_dataset) * 0.8)
valid_set_size = len(train_dataset) - train_set_size
train_set, val_set = random_split(
    train_dataset,
    [train_set_size, valid_set_size],
    generator=torch.Generator().manual_seed(42),
)

batch_size = 128
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = ConvNet()
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
trainer = Dojo(
    model=model,
    loss_function=loss_fn,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    device=device,
)
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--name", type=str, default="")
args = parser.parse_args()
epochs = args.epochs
name = args.name
# epochs = 100
# name = "work18"

epoch_train_losses = []
epoch_val_losses = []
epoch_train_accs = []
epoch_val_accs = []
# scheduler = StepLR(optimizer, step_size=25, gamma=0.999)
with open(f"{name}.txt", "w") as file_write:
    for e in range(epochs):
        file_write.write(f"\nEpoch {e + 1}\n-------------------------------")
        # print(f"Epoch {e + 1}\n-------------------------------")
        train_loss, val_loss, train_acc, val_acc = trainer.train()
        # scheduler.step()
        file_write.write("\nTrain loss at epoch {}: {}".format(e + 1, train_loss))
        # print("\nTrain loss at epoch {}: {}".format(e, train_loss))
        # print("Val loss at epoch {}: {}".format(e, val_loss))
        file_write.write("\nVal loss at epoch {}: {}".format(e + 1, val_loss))
        file_write.write("\n\nTrain acc at epoch {}: {}".format(e + 1, train_acc))
        file_write.write("\nVal acc at epoch {}: {}".format(e + 1, val_acc))
        # print("\nTrain acc at epoch {}: {}".format(e, train_acc))
        # print("Val acc at epoch {}: {}".format(e, val_acc))
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_accs.append(val_acc)

    acc_test = trainer.test()
    file_write.write(
        f"\n-------------------------------\nTest Accuracy of the model: {acc_test * 100:.2f}"
    )
    # print(
    #     f"\n-------------------------------\nTest Accuracy of the model: {acc_test * 100:.2f}"
    # )
torch.save(model, f"CnnMnist{epochs}-{name}.pt")

create_plot(
    epoch_train_losses,
    epoch_val_losses,
    epoch_train_accs,
    epoch_val_accs,
    name,
    epochs,
    save_plot=True,
    display_plot=False,
)
