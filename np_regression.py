import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NP(nn.Module):
    def __init__(self, hdim, ddim, z_samples):
        super(NP, self).__init__()
        in_dim = 2
        out_dim = 2
        self.z_dim = 2
        self.z_samples = z_samples

        self.h1 = nn.Linear(in_dim, hdim)
        self.h2 = nn.Linear(hdim, out_dim)

        self.r_to_z_mean = nn.Linear(in_dim, 1)
        self.r_to_z_std = nn.Linear(in_dim, 1)

        self.d1 = nn.Linear(in_dim + 1, ddim)
        self.d2 = nn.Linear(ddim, out_dim)

        nn.init.normal(self.h1.weight)
        nn.init.normal(self.h2.weight)
        nn.init.normal(self.d1.weight)
        nn.init.normal(self.d2.weight)

    def data_to_r(self, x, y):
        x_y = torch.cat([x, y], dim=1)
        r_i = self.h2(F.relu(self.h1(x_y)))

        # mean aggregate
        r = r_i.mean(dim=0)
        return r

    def r_to_z(self, r):
        mean = self.r_to_z_mean(r)
        log_var = self.r_to_z_std(r)
        return mean, F.softplus(log_var)

    def reparametrize(self, mu, std, n):
        eps = torch.autograd.Variable(std.data.new(n, self.z_dim).normal_())
        z = eps.mul(std).add_(mu)
        return z


    def decoder(self, x_pred, z):
        z = z.unsqueeze(-1).expand(z.size(0), z.size(1), x_pred.size(0)).transpose(1, 2)
        x_pred = x_pred.unsqueeze(0).expand(z.size(0), x_pred.size(0), x_pred.size(1))

        x_z = torch.cat([x_pred, z], dim=-1)
        decode = self.d2(F.sigmoid(self.d1(x_z).squeeze(-1).transpose(0, 1)))
        mu, logstd = torch.split(decode, 1, dim=-1)
        mu = mu.squeeze(-1)
        logstd = logstd.squeeze(-1)
        std = F.softplus(logstd)
        return mu, std

    def forward(self, inputs):
        x_context, y_context, x_target, y_taget = inputs
        x_all = torch.cat([x_context, x_target], dim = 0)
        y_all = torch.cat([y_context, y_taget], dim = 0)
        r = self.data_to_r(x_context, y_context)
        z_mean, z_std = self.r_to_z(r)
        r_all = self.data_to_r(x_all, y_all)
        z_mean_all, z_std_all = self.r_to_z(r_all)

        zs = self.reparametrize(z_mean_all, z_std_all, self.z_samples)
        mu, std = self.decoder(x_context, zs)
        return mu, std, z_mean_all, z_std_all, z_mean, z_std

def log_likelihood(mu, std, target):
    norm = torch.distributions.Normal(mu, std)
    return norm.log_prob(target).sum(dim=0).mean()

def KLD_gaussian(mu_q, std_q, mu_p, std_p):
    var_p = std_p**2 + 1e-10
    var_q = std_q**2 + 1e-10
    return (var_q/var_p + ( (mu_q-mu_p)**2) / var_p + torch.log(var_p/var_q) - 1.0).sum() * 0.5

def random_split_c_t(x, y, n_context):
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return [x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)]

def visualize(x, y, x_star, model):
    r_z = model.data_to_r(x, y)
    z_mu, z_std = model.r_to_z(r_z)
    zsamples = model.reparametrize(z_mu, z_std, 3)
    mu, sigma = model.decoder(x_star, zsamples)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(mu.size(1)):
        ax.plot(x_star.data.cpu().numpy(), mu[:, i].data.cpu().numpy(), linewidth=1)
        ax.fill_between(
            x_grid[:, 0].data.cpu().numpy(), (mu[:, i] - sigma[:, i]).detach().cpu().numpy(),
            (mu[:, i] + sigma[:, i]).detach().cpu().numpy(), alpha=0.2
        )
        ax.scatter(x.data.cpu().numpy(), y.data.cpu().numpy(), color='b')
        ax.plot(all_x_np, all_y_np, color='b')
    plt.pause(0.0001)
    fig.canvas.draw()

x_grid = torch.from_numpy(np.arange(-5, 5, .1).reshape(-1, 1).astype(np.float32))

# create dataset
all_x_np = np.arange(-5, 5, .1).reshape(-1, 1).astype(np.float32)
# all_y_np = np.sin(all_x_np)
all_y_np = np.exp(np.cos(all_x_np))**3 * 2*np.sin(all_x_np) - np.sin(all_x_np)*np.cos(all_x_np)

model = NP(8, 8, 20).to(device)
optimizer =optim.Adam(model.parameters(), lr=0.01)

def train(epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = random_split_c_t(all_x_np, all_y_np, np.random.randint(20, 30))
        for i in range(len(inputs)):
            inputs[i] = torch.from_numpy(inputs[i]).to(device)
        mu, std, z_mean_all, z_std_all, z_mean, z_std = model(inputs)
        loss = -log_likelihood(mu, std, inputs[1]) + KLD_gaussian(z_mean_all, z_std_all, z_mean, z_std)
        loss.backward()
        training_loss = loss.item()
        optimizer.step()
        print('epoch: {} loss: {}'.format(epoch, training_loss/200))

        if epoch % 100 == 0:
            visualize(inputs[0], inputs[1],
                      torch.from_numpy(np.arange(-5, 5, .1).reshape(-1, 1).astype(np.float32)), model)

train(90000)