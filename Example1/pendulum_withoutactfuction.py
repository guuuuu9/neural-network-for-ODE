import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(4)

def ode_solve(z0, t0, t1, f):
    """
    RK4
    """
    h = t1 - t0
    k1 = f(z0, t0)
    k2 = f(z0 + k1 * h / 2., t0 + h / 2.)
    k3 = f(z0 + k2 * h / 2., t0 + h / 2.)
    k4 = f(z0 + k3 * h, t0 + h)
    z = z0 + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)    
    return z

class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp"""
        batch_size = z.shape[0]
        out = self.forward(z, t)
        a = grad_outputs
        adfdz, *adfdp = torch.autograd.grad(
            (out,), (z,) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        return out, adfdz, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                # adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp), dim=1)

        grad_output = grad_output.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(grad_output)
            adj_p = torch.zeros(bs, n_params).to(grad_output)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                # f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = grad_output[i_t]
            
                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z)), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients 
            dLdz_0 = grad_output[0]

            # Adjust adjoints
            adj_z += dLdz_0
        return None, None, adj_p, None

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t, return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]

class LinearODEF(ODEF):
    def __init__(self):
        super(LinearODEF, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 2)
        
    def forward(self, x, t):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

if __name__ == '__main__':
    # writer = SummaryWriter('./logstrain2')
    ode_trained = NeuralODE(LinearODEF())
    # def del_tensor_ele_n(arr, index, n):
    #     arr1 = arr[0:index,:]
    #     arr2 = arr[index+n:,:]
    #     return torch.cat((arr1,arr2),dim=0)

    def create_batch(X, t):
        index_np = np.arange(0, 240, 1)
        index_np = np.hstack([index_np[:, None]])
        t0 = np.random.uniform(0, 23.99 - 5.0)
        t1 = t0 + np.random.uniform(1.0, 5.0)
        t_copy = t.numpy()
        idx = sorted(index_np[(t_copy > t0) & (t_copy < t1)][:15])
        X = X.unsqueeze(1)
        t = t.unsqueeze(1)
        batch_targets = X[idx]
        batch_t = t[idx]
        return batch_targets, batch_t
    
    def plot_results(time, X_pred, X):
        time = time.detach().numpy()
        X_pred = X_pred.detach().numpy()
        plt.plot(time, X_pred[:,0])
        plt.plot(time, X[:,0], alpha=0.7)
        plt.title('Pendulum Motion', fontdict={'size': 14})
        plt.xlabel('time (s)', fontdict={'size': 14})
        plt.ylabel(r'$\theta$ (rad)', fontdict={'size': 14})
        plt.xticks(size = 13)
        plt.yticks(size = 13)
        plt.legend([r'$\theta$_predict',r'$\theta$_true'], loc='upper right')
        plt.grid(True)
        plt.show()

    # training starts            
    optimizer = torch.optim.SGD(ode_trained.parameters(), lr=0.01)
    train_step = 0
    traini=pd.read_excel('./output/data{}.xls'.format(100))
    t = torch.tensor(traini['t'].values).unsqueeze(1)
    x1 = torch.tensor(traini['x1'].values).unsqueeze(1)
    x2 = torch.tensor(traini['x2'].values).unsqueeze(1)
    training_data = torch.cat((t, x1, x2), 1)
    t, X = torch.split(training_data, [1,2], dim=1)
    X = X.type(torch.float32) # 0.032761890441179276 (300,2)
    # X = X + torch.randn_like(X) * 0.01 # 0.03739318251609802 0.03764960542321205
    # X = X + torch.randn_like(X) * 0.02 # 0.04622281715273857 0.05198590084910393
    # X = X + torch.randn_like(X) * 0.03 # 0.059001293033361435 0.07005573064088821
    t = t.type(torch.float32) #(300,1)

    # d = random.randint(0,240)
    X_train = X[0:240]
    t_train = t[0:240]

    for _ in range(800):             
        batch_targets, batch_t = create_batch(X_train, t_train)
        # if batch_targets == torch.Size([]) or batch_t == torch.Size([]):
        #     break
        # else:
        X0 = Variable(batch_targets[0])
        pred = ode_trained(X0, batch_t, return_whole_sequence=True)
        loss_f = nn.MSELoss()
        loss = loss_f(pred, batch_targets.detach())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        train_step += 1
        if train_step % 10 == 0:
            print("train step: {}, ".format(train_step) + "train loss: {}".format(loss.item()))
            # writer.add_scalar('train_loss', loss.item(), train_step)
    X_pred = ode_trained(X[0], t, return_whole_sequence=True)
    plot_results(t, X_pred, X)
    print(loss_f(X_pred, X).item())
    # writer.close()