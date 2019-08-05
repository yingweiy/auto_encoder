# modified based on pytorch example VAE
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class VAE_Model(nn.Module):
    def __init__(self):
        super(VAE_Model, self).__init__()
        self.input_dim = 28*28
        self.n_latent = 20
        self.fc1 = nn.Linear(self.input_dim, 400)
        self.fc21 = nn.Linear(400, self.n_latent)
        self.fc22 = nn.Linear(400, self.n_latent)
        self.fc3 = nn.Linear(self.n_latent, 400)
        self.fc4 = nn.Linear(400, self.input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self):
        std = torch.exp(0.5*self.logvar)
        eps = torch.randn_like(std)
        return self.mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        self.mu, self.logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize()
        return self.decode(z)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        return BCE + KLD

class VAE():
    def __init__(self):
        self.batch_size = 128
        self.epochs = 30
        self.seed = 1
        self.log_interval = 10
        self.learning_rate = 1e-3

        torch.manual_seed(self.seed)
        self.device = torch.device("cuda:0")
        self.model = VAE_Model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.get_data_loader()

    def get_data_loader(self):
        kwargs = {'num_workers': 1, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=self.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=self.batch_size, shuffle=True, **kwargs)


    def train_batch(self):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.model.loss(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              self.epoch, train_loss / len(self.train_loader.dataset)))


    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.model.loss(recon_batch, data).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                          recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                             'results/reconstruction_' + str(self.epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def train(self):
        for self.epoch in range(1, self.epochs + 1):
            self.train_batch()
            self.test()
            with torch.no_grad():
                sample = torch.randn(64, self.model.n_latent).to(self.device)
                sample = self.model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28),
                           'results/sample_' + str(self.epoch) + '.png')

if __name__ == "__main__":
    vae = VAE()
    vae.train()
