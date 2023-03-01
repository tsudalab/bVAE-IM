import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab, MolTree
from .nnutils import create_var, flatten_tensor, avg_pool
from .jtnn_enc import JTNNEncoder
from .jtnn_dec import JTNNDecoder
from .mpn import MPN
from .jtmpn import JTMPN
from .datautils import tensorize

from .chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols
import rdkit
import rdkit.Chem as Chem
import copy, math
import numpy as np
from tqdm import tqdm

class JTNNVAE(nn.Module):

    def __init__(self, vocab, binary_size, depthT, depthG, n_class=2, device=torch.device("cuda")):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.n_class = n_class
        self.binary_size = binary_size = binary_size // 2

        self.jtnn = JTNNEncoder(binary_size*n_class, depthT, nn.Embedding(vocab.size(), binary_size*n_class), device)
        self.decoder = JTNNDecoder(vocab, binary_size*n_class, binary_size*n_class, nn.Embedding(vocab.size(), binary_size*n_class), device)

        self.jtmpn = JTMPN(binary_size*n_class, depthG, device)
        self.mpn = MPN(binary_size*n_class, depthG, device)

        self.A_assm = nn.Linear(binary_size*n_class, binary_size*n_class, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(reduction='sum')

        self.device = device

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder) # tree_vecs: batch x binary_size*n_class
        logits_tree_vecs = tree_vecs.view(-1, self.n_class) # logits_tree_vecs: batch*binary_size x n_class
        q_tree_vecs = F.softmax(logits_tree_vecs, dim=-1).view(-1, self.binary_size*self.n_class)
        
        mol_vecs = self.mpn(*mpn_holder) # mol_vecs: batch x binary_size*n_class)
        logits_mol_vecs = mol_vecs.view(-1, self.n_class) # logits_mol_vecs: batch*binary_size x n_class)
        q_mol_vecs = F.softmax(logits_mol_vecs, dim=-1).view(-1, self.binary_size*self.n_class)
        return q_tree_vecs, logits_tree_vecs, tree_mess, q_mol_vecs, logits_mol_vecs
    
    def encode_from_smiles(self, smiles_list, n_sample=None):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, logits_tree_vecs, _, mol_vecs, logits_mol_vecs = self.encode(jtenc_holder, mpn_holder)
        if n_sample is not None:
            temp = 0.4
            tree_vecs = tree_vecs.repeat_interleave(n_sample, dim=0)
            logits_tree_vecs = logits_tree_vecs.repeat_interleave(n_sample, dim=0)
            tree_vecs, _ = self.gumbel_softmax(tree_vecs, logits_tree_vecs, temp=temp)

            mol_vecs = mol_vecs.repeat_interleave(n_sample, dim=0)
            logits_mol_vecs = logits_mol_vecs.repeat_interleave(n_sample, dim=0)
            mol_vecs, _ = self.gumbel_softmax(mol_vecs, logits_mol_vecs, temp=temp)
         
        tree_vecs = torch.reshape(tree_vecs, (-1, self.binary_size, 2))
        mol_vecs = torch.reshape(mol_vecs, (-1, self.binary_size, 2))
        return torch.cat([torch.argmax(tree_vecs, dim=-1), torch.argmax(mol_vecs, dim=-1)], dim=-1)

    def gumbel_softmax(self, q, logits, temp):
        G_sample = self.gumbel_sample(logits.shape).to(self.device)
        y = F.softmax((logits + G_sample) / temp, dim=-1).view(-1, self.binary_size*self.n_class)
        kl_loss = torch.sum(q * torch.log(q * self.n_class + 1e-20), dim=-1).mean().to(self.device)
        return y, kl_loss

    def gumbel_sample(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def sample_prior(self, prob_decode=False, return_vec=False):
        bi_sample = np.zeros((self.binary_size*2, self.n_class), dtype=np.float32)
        bi_sample[range(self.binary_size*2), np.random.choice(self.n_class, self.binary_size*2)] = 1
        bi_vec = torch.from_numpy(bi_sample.argmax(1))
        bi_sample = np.reshape(bi_sample, [self.binary_size*2, self.n_class])
        
        tree = torch.from_numpy(bi_sample[:self.binary_size,:]).view(1, -1).to(self.device)
        mol = torch.from_numpy(bi_sample[self.binary_size:,:]).view(1, -1).to(self.device)
        if not return_vec:
            return self.decode(tree, mol, prob_decode)
        return bi_vec, self.decode(tree, mol, prob_decode)

    def forward(self, x_batch, beta, temp):
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        q_tree_vecs, logits_tree_vecs, tree_mess, q_mol_vecs, logits_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        g_tree_vecs, tree_kl = self.gumbel_softmax(q_tree_vecs, logits_tree_vecs, temp)
        g_mol_vecs, mol_kl = self.gumbel_softmax(q_mol_vecs, logits_mol_vecs, temp)

        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, g_tree_vecs)
        assm_loss, assm_acc = self.assm(x_batch, x_jtmpn_holder, g_mol_vecs, tree_mess)

        return word_loss + topo_loss + assm_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder,batch_idx = jtmpn_holder
        fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
        batch_idx = batch_idx.to(self.device)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs) #bilinear
        scores = torch.bmm(
                x_mol_vecs.unsqueeze(1),
                cand_vecs.unsqueeze(-1)
        ).squeeze()
        
        cnt,tot,acc = 0,0,0
        all_loss = []
        for i,mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = torch.LongTensor([label]).to(self.device)
                all_loss.append( self.assm_loss(cur_score.view(1,-1), label) )
        
        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        #currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root,pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.smiles

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict) #Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze() #bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol,_ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=True)
        if cur_mol is None: 
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol,pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None: 
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
        
    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles,cand_amap = list(zip(*cands))
        aroma_score = torch.Tensor(aroma_score).to(self.device)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            probs = F.softmax(scores.view(1,-1), dim=1).squeeze() + 1e-7 #prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None: 
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol

    def decode_from_binary(self, binary, prob_decode=False):
        binary = F.one_hot(binary.long(),num_classes=2).float()
        binary = binary.view(1, -1)
        tree_vec = binary[:, :self.binary_size*2]
        mol_vec = binary[:, self.binary_size*2:]
        smiles = self.decode(tree_vec, mol_vec, prob_decode)
        return smiles

    def decode_from_binaries(self, binary_set, prob_decode=False):
        smiles_set = []
        for binary in tqdm(binary_set):
            binary = F.one_hot(binary.long(),num_classes=2).float()
            binary = binary.view(1, -1)
            tree_vec = binary[:, :self.binary_size*2]
            mol_vec = binary[:, self.binary_size*2:]
            smiles = self.decode(tree_vec, mol_vec, prob_decode)
            smiles_set.append(smiles)
        return smiles_set
