import sys, os
sys.path.append('../')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import random, sys
import numpy as np

import pickle as pickle

from model import *
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')   
from rdkit import Chem
from sklearn.model_selection import train_test_split

from amplify import gen_symbols, BinaryPoly, sum_poly
from amplify import decode_solution, Solver
from amplify.client import FixstarsClient
from amplify.client.ocean import DWaveSamplerClient

from tqdm import tqdm

# factorization machine
class TorchFM(nn.Module):
    
    def __init__(self, n=None, k=None):
        # n: size of binary features
        # k: size of latent features
        super().__init__()
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        out = out.squeeze(dim=1)

        return out


class MolData(Dataset):
    
    def __init__(self, binary, targets):
        self.binary = binary
        self.targets = targets

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, index):
        return self.binary[index], self.targets[index]



class bVAE_IM(object):
    def __init__(self,
        train_smiles,
        train_prop,
        bvae_model,
        score_function,
        opt_target='max',
        random_seed = None,
        device = torch.device("cuda")):

        self.train_smiles = np.load(train_smiles)
        self.train_targets = np.load(train_prop)
        self.bvae_model = bvae_model
        self.get_score = score_function
        if opt_target not in ['max', 'min']:
            raise ValueError("opt_target should be max or min.")
        self.opt_target = opt_target
        self.device = device

        self.random_seed = random_seed
        if random_seed is not None:
            seed_all(random_seed)

        self._initialize()
        self.n_binary = self.train_binary.shape[1]


    def optimize(self,
        result_save_dir,
        fm_save_dir,
        client,
        client_token,
        n_factor = 8,
        n_opt = 300,
        fm_train_batch = 64,
        fm_max_epoch = 10000,
        fm_patience = 500,
        fm_learning_rate = 0.001,
        fm_weight_decay_rate = 0.01,
        ):

        self.client = client

        self.results_smiles = []
        self.results_binary = []
        self.results_scores = []

        fm_model = TorchFM(self.n_binary, n_factor).to(self.device)

        # config Ising machine

        if client == "amplify":
            client = FixstarsClient()
            client.token = client_token
            client.parameters.timeout = 1000
        elif client == "dwave":
            client = DWaveSamplerClient()
            client.token = client_token
            client.solver = 'Advantage_system4.1'
            client.parameters.num_reads = 1000
            client.parameters.max_answers = 1
        else:
            raise TypeError("Client not supported!")

        solver = Solver(client)

        self.iteration = 0

        while self.iteration < n_opt:
            for param in fm_model.parameters():
                if param.dim() == 1:
                    nn.init.constant_(param, 0)     # bias
                else:
                    nn.init.uniform_(param, -0.03, 0.03)   # weights
            
            # train factorization machine
            fm_model = self._train_fm(model = fm_model,
                                    model_save_dir = fm_save_dir,
                                    batch_size = fm_train_batch,
                                    max_epoch = fm_max_epoch,
                                    patience = fm_patience,
                                    learning_rate = fm_learning_rate,
                                    weight_decay_rate = fm_weight_decay_rate,
                                    split_random_seed = self.iteration
                                    )
            # solve QUBO
            solution, energy = self._solve_qubo(fm_model = fm_model,
                                    n_factor = n_factor,
                                    qubo_solver = solver
                                    )
            # merge new data into dataset
            self._update(solution=solution,
                        energy=energy,
                        fm_model=fm_model
                        )

        
        if not os.path.exists(result_save_dir):
            os.mkdir(result_save_dir)
        
        with open((os.path.join(result_save_dir, "logp_smiles.pkl")), "wb") as f:
            pickle.dump(self.results_smiles, f)
        with open((os.path.join(result_save_dir, "logp_scores.pkl")), "wb") as f:
            pickle.dump(self.results_scores, f)
        

    def _initialize(self):
        self.train_smiles = self.train_smiles.tolist()
        self.train_targets = self.train_targets.astype('float')
        self.train_mols = [Chem.MolFromSmiles(s) for s in self.train_smiles]

        self.train_binary = self._encode_to_binary(self.train_smiles)

        if self.opt_target == 'max':
            self.train_targets = [-self.get_score(m) for m in self.train_mols]
        elif self.opt_target == 'min':
            self.train_targets = [self.get_score(m) for m in self.train_mols]
        # plus --> minimization; minus --> maximization

    def _encode_to_binary(self, smiles, batch_size = 64):
        encoded = []
        print("encoding molecules to binary sequences...")
        for i in tqdm(range(int(np.ceil(len(smiles) / batch_size)))):
            smiles_batch = smiles[i*batch_size: (i+1)*batch_size]
            encoded_batch = self.bvae_model.encode_from_smiles(smiles_batch)
            encoded.append(encoded_batch)
        train_binary = torch.vstack(encoded)
        train_binary = train_binary.to('cpu').numpy()
        return train_binary

    def _train_fm(self,
        model,
        model_save_dir,
        batch_size = 64,
        max_epoch = 10000,
        patience = 500,
        learning_rate = 0.001,
        weight_decay_rate = 0.01,
        split_random_seed = None):


        X_train, X_valid, y_train, y_valid = train_test_split(self.train_binary,
                                                            self.train_targets,
                                                            test_size=0.1,
                                                            random_state=split_random_seed)
        # X_train = X_train.to(torch.float)
        # X_valid = X_valid.to(torch.float)
        X_train = torch.from_numpy(X_train).to(torch.float).to(self.device)
        X_valid = torch.from_numpy(X_valid).to(torch.float).to(self.device)
        y_train = torch.tensor(y_train).to(torch.float).to(self.device)
        y_valid = torch.tensor(y_valid).to(torch.float).to(self.device)

        dataset_train = MolData(X_train, y_train)
        dataloader_train = DataLoader(dataset=dataset_train,
                                    batch_size=batch_size,
                                    shuffle=True)
        dataset_valid = MolData(X_valid, y_valid)
        dataloader_valid = DataLoader(dataset=dataset_valid,
                                    batch_size=batch_size,
                                    shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
        criterion = nn.MSELoss()

        lowest_error = float('inf')
        best_epoch = 0

        for epoch in range(max_epoch):
            model.train()
            for batch_x, batch_y in dataloader_train:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_hat_valid = []
                for batch_x, _ in dataloader_valid:
                    valid = model(batch_x)
                    y_hat_valid.append(valid)
                y_hat_valid = torch.concat(y_hat_valid)

                epoch_error = criterion(y_valid, y_hat_valid)
                epoch_error = epoch_error.detach().cpu().numpy()
                if epoch % 100 == 0:
                    print("Model -- Epoch %d error on validation set: %.4f" % (epoch, epoch_error))
                
                if epoch_error < lowest_error:
                    torch.save(model.state_dict(), os.path.join(model_save_dir, "fm_model-logp-%s-dim%d-seed%d" % (self.client, self.n_binary, self.random_seed)))
                    lowest_error = epoch_error
                    best_epoch = epoch

                if epoch > best_epoch+patience:
                    print("Model -- Epoch %d has lowest error!" % (best_epoch))
                    break
        
        y_hat_valid = y_hat_valid.detach().cpu().numpy()
        y_valid = y_valid.detach().cpu().numpy()

        # reload best epoch
        model.load_state_dict(torch.load(os.path.join(model_save_dir, "fm_model-logp-%s-dim%d-seed%d" % (self.client, self.n_binary, self.random_seed))))

        return model


    def _solve_qubo(self,
        fm_model,
        n_factor,
        qubo_solver
        ):

        # extract parameters from FM for building the QUBO graph
        for p in fm_model.parameters():
            if tuple(p.shape) == (self.n_binary, n_factor):
                Vi_f= p.to("cpu").detach().numpy()
            elif tuple(p.shape) == (1, self.n_binary):
                Wi = p.to("cpu").detach().numpy()
            elif tuple(p.shape) == (1, ):
                W0 = p.to("cpu").detach().numpy()

        # build the QUBO graph
        q = gen_symbols(BinaryPoly, self.n_binary)
        f_E = sum_poly(n_factor, lambda f: ((sum_poly(self.n_binary, lambda i: Vi_f[i][f] * q[i]))**2 - sum_poly(self.n_binary, lambda i: Vi_f[i][f]**2 * q[i]**2)))/2 \
            + sum_poly(self.n_binary, lambda i: Wi[0][i]*q[i]) \
            + W0[0]

        # solve QUBO
        solver = qubo_solver
        result = solver.solve(f_E)

        sols = []
        sol_E = []
        for sol in result:  # Iterate over multiple solutions
            solution = decode_solution(q, sol.values)  #  Decode variable array q with sol.values
            sols.append(solution)
            sol_E.append(sol.energy)
        return np.array(sols), np.array(sol_E).astype(np.float32)

    def _update(self,
        solution,
        energy,
        fm_model):

        self.iteration += 1

        binary_new = torch.from_numpy(solution).to(torch.float).to(self.device)
        smiles_new = self.bvae_model.decode_from_binary(binary_new)   # 1 x 1
        mol_new = Chem.MolFromSmiles(smiles_new)

        # skip invalid smiles
        if mol_new is None:
            return

        if smiles_new in self.train_smiles:
            return

        # fm_pred = fm_model(binary_new)
        # fm_pred = fm_pred.detach().cpu().numpy()

        # assert np.round(fm_pred, 3) == np.round(energy[0], 3)    # ensure correctness of qubo
        if self.opt_target == 'max':
            target_new = -self.get_score(mol_new)
        else:
            target_new = self.get_score(mol_new)

        self.train_smiles.append(smiles_new)
        binary_new = binary_new.to('cpu').numpy()
        self.train_binary = np.vstack((self.train_binary, binary_new))
        self.train_targets.append(target_new)

        self.results_smiles.append(smiles_new)
        self.results_binary.extend(solution)
        self.results_scores.append(-target_new)
        
        assert self.train_binary.shape[0] == len(self.train_targets)

        return


def main(model_path,
        vocab_path,
        binary_dim,
        train_smiles,
        train_prop,
        output_path,
        cache_path,
        token,
        depthT=20,
        depthG=3,
        factor_num=8,
        fm_decay_weight=0.01,
        fm_max_epoch=10000,
        fm_lr=1e-4,
        fm_patience=500,
        batch_size=64,
        opt_target='max',
        num_end=300,
        seed=None,
        device='cuda',
        client='amplify'):

    vocab = [x.strip("\r\n ") for x in open(vocab_path)] 
    vocab = Vocab(vocab)

    vae_model = JTNNVAE(vocab=vocab,
        binary_size=binary_dim,
        depthT=depthT,
        depthG=depthG,
        device=device)
    vae_model = vae_model.to(device)
    vae_model.load_state_dict(torch.load(model_path))

    optimizer = bVAE_IM(train_smiles=train_smiles,
                        train_prop=train_prop,
                        bvae_model=vae_model,
                        score_function=score_function,
                        opt_target=opt_target,
                        random_seed=seed,
                        device=device)

    optimizer.optimize(result_save_dir=output_path,
                fm_save_dir=cache_path,
                client=client,
                client_token=token,
                n_factor=factor_num,
                n_opt=num_end,
                fm_train_batch=batch_size,
                fm_max_epoch=fm_max_epoch,
                fm_patience=fm_patience,
                fm_learning_rate=fm_lr,
                fm_weight_decay_rate=fm_decay_weight)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_path', help='model path', required=True, type=str)
    parser.add_argument('--vocab', dest='vocab_path', help='vocab path', required=True, type=str)
    parser.add_argument('--dim', dest='binary_dim', help='binary dimension', required=True, type=int)
    parser.add_argument('--smiles', dest='train_smiles', help='smiles training data for factorization machine', required=True, type=str)
    parser.add_argument('--prop', dest='train_prop', help='property training data for factorization machine', required=True, type=str)

    parser.add_argument('--output', dest='output_path', help='output path', required=True, type=str)
    parser.add_argument('--cache', dest='cache_path', help='cache path', required=True, type=str)
    
    parser.add_argument('--token', dest='token', help='client token', required=True, type=str)

    parser.add_argument('--depthT', dest='depthT', help='tree depth', default=20, type=int)
    parser.add_argument('--depthG', dest='depthG', help='graph depth', default=3, type=int)

    parser.add_argument('--factor', dest='factor_num', help='factorization size', default=8, type=int)
    parser.add_argument('--decay', dest='decay_weight', help='decay weight', default=0.01, type=float)
    parser.add_argument('--maxepoch', dest='max_epoch', help='max epoch', default=10000, type=int)
    parser.add_argument('--lr', dest='learning_rate', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--patience', dest='patience', help='training patience', default=500, type=int)
    parser.add_argument('--batch', dest='batch_size', help='batch size for factorization machine', default=64, type=int)

    parser.add_argument('--target', dest='opt_target', help='maximize or minimize the target score', default='max', type=str)
    parser.add_argument('--num', dest='num_end', help='number for end condition', default=300, type=int)
    parser.add_argument('--seed', dest='random_seed', help='random seed', default=0, type=int)
    parser.add_argument('--device', dest='device', help='cpu or cuda', default='cuda', type=str)
    parser.add_argument('--client', dest='client', help='amplify or dwave', default='amplify', type=str)

    args = parser.parse_args()

    from scorers.logp_scores import score_function

    main(model_path=args.model_path,
        vocab_path=args.vocab_path,
        binary_dim=args.binary_dim,
        train_smiles=args.train_smiles,
        train_prop=args.train_prop,
        output_path=args.output_path,
        cache_path=args.cache_path,
        token=args.token,
        depthT=args.depthT,
        depthG=args.depthG,
        factor_num=args.factor_num,
        fm_decay_weight=args.decay_weight,
        fm_max_epoch=args.max_epoch,
        fm_lr=args.learning_rate,
        fm_patience=args.patience,
        batch_size=args.batch_size,
        opt_target=args.opt_target,
        num_end=args.num_end,
        seed=args.random_seed,
        device=args.device,
        client=args.client
    )
