import sys, os
sys.path.append('../')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import random, sys
import numpy as np
import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())
import pickle as pickle

from bJTVAE import *

from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')   
from rdkit import Chem
from sklearn.model_selection import train_test_split

from amplify import BinaryMatrix, BinaryPoly, gen_symbols, sum_poly
from amplify import decode_solution, Solver
from amplify.client import FixstarsClient
from amplify.client.ocean import DWaveSamplerClient

from sklearn.linear_model import Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import logging
import time

UPDATE_ITER = 1

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


class RandomBinaryData(Dataset):

    def __init__(self, binary):
        self.binary = binary

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, index):
        return self.binary[index]


class bVAE_IM(object):
    def __init__(self, bvae_model, configs, device = torch.device("cuda")):

        self.train_smiles = np.load(configs['opt']['train_smiles'])
        self.train_targets = np.load(configs['opt']['train_prop'])
        self.bvae_model = bvae_model

        target_prop = configs['opt']['prop']
        if target_prop == 'logp':
            from scorers.logp_scores import score_function
        elif target_prop == 'tpsa':
            from scorers.tpsa_scores import score_function
        elif target_prop == 'multi':
            from scorers.multi_scores import score_function
        elif target_prop == 'mw':
            from scorers.mw_scores import score_function
        elif target_prop == 'aroring':
            from scorers.aroring_scores import score_function
        elif target_prop == 'rotbond':
            from scorers.rotbond_scores import score_function
        else:
            raise ValueError("please define the score function first.")
        self.get_score = score_function

        if configs['opt']['target'] not in ['max', 'min']:
            raise ValueError("opt target should be max or min.")
        self.opt_target = configs['opt']['target']
        self.device = device

        self.random_seed = configs['seed']
        if self.random_seed is not None:
            seed_all(self.random_seed)
        
        self.n_sample = configs['opt']['n_sample']
        self._initialize()
        self.n_binary = self.train_binary.shape[1]
        self.sleep_count = 0


    def optimize(self, configs):
        
        n_opt = configs['opt']['num_end']
        
        self.end_cond = configs['opt']['end_cond']
        if self.end_cond not in [0, 1, 2]:
            raise ValueError("end_cond should be 0, 1 or 2.")
        if self.end_cond == 2:
            n_opt = 100 # n_opt is patience in this condition. When patience exceeds 100, exhaustion searching ends.

        self.results_smiles = []
        self.results_binary = []
        self.results_scores = []

        # # config Ising machine

        client = configs['opt']['client']
        if client == "amplify":
            client = FixstarsClient()
            client.token = configs['opt']['client_token']
            client.parameters.timeout = 1000
        elif client == "dwave":
            client = DWaveSamplerClient()
            client.token = configs['opt']['client_token']
            client.solver = configs['opt']['dwave_sys']
            client.parameters.num_reads = 1000
            client.parameters.max_answers = 1
        else:
            raise ValueError("Wrong client!")

        solver = Solver(client)

        self.iteration = 0

        while self.iteration < n_opt:

            # train factorization machine
            qubo = self._build_qubo(configs)

            solution, energy = self._solve_qubo(qubo = qubo,
                                    qubo_solver = solver)

            # merge new data into dataset
            self._update(solution=solution,
                        energy=energy)

        result_save_dir = configs['opt']['output']
        if not os.path.exists(result_save_dir):
            os.mkdir(result_save_dir)
        
        with open((os.path.join(result_save_dir, "%s_smiles.pkl" % configs['opt']['prop'])), "wb") as f:
            pickle.dump(self.results_smiles, f)
        with open((os.path.join(result_save_dir, "%s_scores.pkl" % configs['opt']['prop'])), "wb") as f:
            pickle.dump(self.results_scores, f)
        
        logging.info("Sleeped for %d minutes..." % self.sleep_count)
        

    def _initialize(self):
        self.train_smiles = self.train_smiles.tolist()
        self.train_targets = self.train_targets.astype('float')
        self.train_mols = [Chem.MolFromSmiles(s) for s in self.train_smiles]

        self.train_binary = self._encode_to_binary(self.train_smiles)

        if self.opt_target == 'max':
            self.train_targets = [-self.get_score(m) for m in self.train_mols]
        elif self.opt_target == 'min':
            self.train_targets = [self.get_score(m) for m in self.train_mols]
        self.train_targets = np.repeat(self.train_targets, self.n_sample).tolist()
        # plus --> minimization; minus --> maximization

    def _encode_to_binary(self, smiles, batch_size = 64):
        encoded = []
        print("encoding molecules to binary sequences...")
        for i in tqdm(range(int(np.ceil(len(smiles) / batch_size)))):
            smiles_batch = smiles[i*batch_size: (i+1)*batch_size]
            if self.n_sample == 1:
                encoded_batch = self.bvae_model.encode_from_smiles(smiles_batch)
            else:
                encoded_batch = self.bvae_model.encode_from_smiles(smiles_batch, self.n_sample)
            encoded.append(encoded_batch)
        train_binary = torch.vstack(encoded)
        train_binary = train_binary.to('cpu').numpy()
        return train_binary

    def _build_qubo(self, configs):

        model_type = configs['opt']['surro_model']

        if model_type in ['ridge', 'lasso', 'pls']:
            if model_type == 'ridge':
                model = Ridge(configs['opt']['decay_weight'])
            elif model_type == 'lasso':
                model = Lasso(configs['opt']['decay_weight'])
            elif model_type == "pls":
                model = PLSRegression(configs['opt']['n_comp'])
            else:
                raise ValueError()
            
            X_train, X_valid, y_train, y_valid = train_test_split(self.train_binary, self.train_targets)

            quad = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            x_train_quad = quad.fit_transform(X_train)
            x_valid_quad = quad.transform(X_valid)
            assert quad.powers_.shape == (self.n_binary + self.n_binary * (self.n_binary-1) // 2, self.n_binary)
            # print(quad.powers_.shape)

            Q_mat = BinaryMatrix(self.n_binary)

            powers = quad.powers_
            model.fit(x_train_quad, y_train)
            score = model.score(X=x_valid_quad, y=y_valid)
            print(score)
            weights = model.coef_

            for i, power in enumerate(powers):
                
                idx = np.argwhere(power == 1)
                if len(idx) == 1:
                    idx = np.squeeze(idx)
                    Q_mat[idx, idx] = weights[i]
                elif len(idx) == 2:
                    idx = np.squeeze(idx)
                    Q_mat[idx[0], idx[1]] = weights[i]
                else:
                    raise ValueError()
                
            qubo = Q_mat
                
        elif model_type == 'fm:
            model = TorchFM(self.n_binary, configs['opt']['factor_num']).to(self.device)
            for param in model.parameters():
                if param.dim() == 1:
                    nn.init.constant_(param, 0)     # bias
                else:
                    nn.init.uniform_(param, -configs['opt']['param_init'], configs['opt']['param_init'])   # weights

            X_train, X_valid, y_train, y_valid = train_test_split(self.train_binary,
                                                            self.train_targets,
                                                            test_size=0.1,
                                                            random_state=self.iteration)

            X_train = torch.from_numpy(X_train).to(torch.float).to(self.device)
            X_valid = torch.from_numpy(X_valid).to(torch.float).to(self.device)
            y_train = torch.tensor(y_train).to(torch.float).to(self.device)
            y_valid = torch.tensor(y_valid).to(torch.float).to(self.device)

            dataset_train = MolData(X_train, y_train)
            dataloader_train = DataLoader(dataset=dataset_train,
                                        batch_size=configs['opt']['batch_size'],
                                        shuffle=True)
            dataset_valid = MolData(X_valid, y_valid)
            dataloader_valid = DataLoader(dataset=dataset_valid,
                                        batch_size=configs['opt']['batch_size'],
                                        shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=configs['opt']['lr'],
                                         weight_decay=configs['opt']['decay_weight'])
            criterion = nn.MSELoss()

            lowest_error = float('inf')
            best_epoch = 0

            for epoch in range(configs['opt']['maxepoch']):
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
                    y_hat_valid = torch.cat(y_hat_valid)

                    epoch_error = criterion(y_valid, y_hat_valid)
                    epoch_error = epoch_error.detach().cpu().numpy()
                    if epoch % 100 == 0:
                        print("Model -- Epoch %d error on validation set: %.4f" % (epoch, epoch_error))
                    
                    if epoch_error < lowest_error:
                        torch.save(model.state_dict(),
                                   os.path.join(configs['opt']['cache'],
                                   "fm_model-%s-%s-dim%d-seed%d-end%d" % (
                                        configs['opt']['prop'],
                                        configs['opt']['client'],
                                        self.n_binary,
                                        self.random_seed,
                                        self.end_cond)))
                        lowest_error = epoch_error
                        best_epoch = epoch

                    if epoch > best_epoch+configs['opt']['patience']:
                        print("Model -- Epoch %d has lowest error!" % (best_epoch))
                        break
            
            y_hat_valid = y_hat_valid.detach().cpu().numpy()
            y_valid = y_valid.detach().cpu().numpy()
            print(np.corrcoef(y_hat_valid, y_valid))
            # reload best epoch
            model.load_state_dict(torch.load(
                os.path.join(configs['opt']['cache'],
                "fm_model-%s-%s-dim%d-seed%d-end%d" % (
                    configs['opt']['prop'],
                    configs['opt']['client'],
                    self.n_binary,
                    self.random_seed,
                    self.end_cond)))
                )

            for p in model.parameters():
                if tuple(p.shape) == (self.n_binary, configs['opt']['factor_num']):
                    Vi_f= p.to("cpu").detach().numpy()
                elif tuple(p.shape) == (1, self.n_binary):
                    Wi = p.to("cpu").detach().numpy()
                elif tuple(p.shape) == (1, ):
                    W0 = p.to("cpu").detach().numpy()

            # build the QUBO graph
            q = gen_symbols(BinaryPoly, self.n_binary)
            f_E = sum_poly(configs['opt']['factor_num'], lambda f: ((sum_poly(self.n_binary, lambda i: Vi_f[i][f] * q[i]))**2 - sum_poly(self.n_binary, lambda i: Vi_f[i][f]**2 * q[i]**2)))/2 \
                + sum_poly(self.n_binary, lambda i: Wi[0][i]*q[i]) \
                + W0[0]
            qubo = (q, f_E)

        return qubo


    def _solve_qubo(self,
        qubo,
        qubo_solver):

        if isinstance(qubo, tuple):
            q, qubo = qubo

        solved = False
        while not solved:
            try:
                result = qubo_solver.solve(qubo)
                solved = True
            except RuntimeError as e: # retry after 60s if connection to the solver fails..
                time.sleep(60)
                self.sleep_count += 1

        sols = []
        sol_E = []
        for sol in result:  # Iterate over multiple solutions
            # solution = [sol.values[i] for i in range(self.n_binary)]
            if isinstance(qubo, BinaryMatrix):
                solution = [sol.values[i] for i in range(self.n_binary)]
            elif isinstance(qubo, BinaryPoly):
                solution = decode_solution(q, sol.values)
            else:
                raise ValueError("qubo type unknown!")
            sols.append(solution)
            sol_E.append(sol.energy)
        return np.array(sols), np.array(sol_E).astype(np.float32)

    def _update(self,
        solution,
        energy):

        # 0 --> certain number of iterations;
        # 1 --> certain number of new molecule;
        # 2 --> exhaustion
        if self.end_cond == 0:
            self.iteration += 1

        binary_new = torch.from_numpy(solution).to(torch.float).to(self.device)
        smiles_new = self.bvae_model.decode_from_binary(binary_new)   # 1 x 1
        mol_new = Chem.MolFromSmiles(smiles_new)

        # skip invalid smiles
        if mol_new is None:
            return

        if smiles_new in self.train_smiles:
            if self.end_cond == 2:
                self.iteration += 1
            return

        # fm_pred = fm_model(binary_new)
        # fm_pred = fm_pred.detach().cpu().numpy()

        # assert np.round(fm_pred, 3) == np.round(energy[0], 3)    # ensure correctness of qubo
        if self.opt_target == 'max':
            target_new = -self.get_score(mol_new)
        else:
            target_new = self.get_score(mol_new)
        print("energy: %.3f; target: %.3f" % (energy[0], target_new))
        self.train_smiles.append(smiles_new)
        # self.train_binary = torch.vstack((self.train_binary, binary_new))
        binary_new = binary_new.to('cpu').numpy()
        self.train_binary = np.vstack((self.train_binary, binary_new))
        print(self.train_binary.shape)
        self.train_targets.append(target_new)


        # if new molecule is generated:
        if self.end_cond == 1:
            self.iteration += 1
        if self.end_cond == 2:
            self.iteration = 0 # if new molecule is generated, reset to 0

        self.results_smiles.append(smiles_new)
        self.results_binary.extend(solution)
        self.results_scores.append(-target_new)

        logging.info("Iteration %d: QUBO energy -- %.4f, actual energy -- %.4f, smiles -- %s" % (self.iteration, energy[0], target_new, smiles_new))
        
        assert self.train_binary.shape[0] == len(self.train_targets)

        return


def main(configs):

    device = torch.device(configs['device'])

    vocab = [x.strip("\r\n ") for x in open(configs['vae_model']['vocab_path'])] 
    vocab = Vocab(vocab)

    vae_model = JTNNVAE(vocab=vocab,
        binary_size=configs['vae_model']['binary_dim'],
        depthT=configs['vae_model']['depthT'],
        depthG=configs['vae_model']['depthG'],
        device=device)
    vae_model = vae_model.to(device)
    vae_model.load_state_dict(torch.load(configs['vae_model']['model_path']))

    optimizer = bVAE_IM(bvae_model=vae_model,
                        configs=configs,
                        device=device)
    
    start_time = time.time()

    optimizer.optimize(configs)

    logging.info("Running Time: %f" % (time.time() - start_time))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', dest='yaml_path', help='yaml path', required=True, type=str)
    args = parser.parse_args()

    with open(args.yaml_path,'r') as f:
        configs = yaml.safe_load(f)

    logging.basicConfig(filename='bVAE-QUBO_%s_%s-dim%d-end%d-num%d-seed%d-%s.log' %\
                         (configs['opt']['prop'],
                          configs['opt']['client'],
                          configs['vae_model']['binary_dim'],
                          configs['opt']['end_cond'],
                          configs['opt']['num_end'], 
                          configs['seed'],
                          configs['opt']['surro_model']),
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.INFO)

    main(configs)
