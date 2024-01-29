import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models.hyperSE import HyperSE
from geoopt.optim import RiemannianAdam
from utils.eval_utils import decoding_cluster_from_tree, cluster_metrics
from utils.plot_utils import plot_leaves, plot_nx_graph
from utils.decode import construct_tree, to_networkx_tree
from dataset import load_data
from utils.train_utils import EarlyStopping
from logger import create_logger
from manifold.poincare import Poincare
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def send_device(self, data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        data = load_data(self.configs)
        self.send_device(data)

        total_nmi = []
        total_ari = []
        for exp_iter in range(self.configs.exp_iters):
            best_cluster_result = {}
            best_cluster = {'acc': 0, 'nmi': 0, 'f1': 0, 'ari': 0}
            logger.info(f"\ntrain iters {exp_iter}")
            model = HyperSE(in_features=data['num_features'],
                            hidden_features=self.configs.hidden_dim,
                            hidden_dim_enc=self.configs.hidden_dim_enc,
                            num_nodes=data['num_nodes'],
                            height=self.configs.height, temperature=self.configs.temperature,
                            embed_dim=self.configs.embed_dim, dropout=self.configs.dropout,
                            nonlin=self.configs.nonlin,
                            decay_rate=self.configs.decay_rate).to(device)
            optimizer = RiemannianAdam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

            pretrained = True
            if pretrained:
                # model.load_state_dict(torch.load(f'checkpoints/{self.configs.save_path}'))
                optimizer_pre = RiemannianAdam(model.parameters(), lr=self.configs.lr_pre, weight_decay=self.configs.w_decay)
                for epoch in range(1, self.configs.pre_epochs):
                    model.train()
                    loss = model.loss(data, device, pretrain=True)
                    optimizer_pre.zero_grad()
                    loss.backward()
                    optimizer_pre.step()
                    logger.info(f"Epoch {epoch}: train_loss={loss.item()}")

            logger.info("--------------------------Training Start-------------------------")
            n_cluster_trials = self.configs.n_cluster_trials
            for epoch in range(1, self.configs.epochs + 1):
                model.train()
                loss = model.loss(data, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                logger.info(f"Epoch {epoch}: train_loss={loss.item()}")
                if epoch % self.configs.eval_freq == 0:
                    logger.info("-----------------------Evaluation Start---------------------")
                    model.eval()
                    embeddings = model(data, device).detach().cpu()
                    manifold = model.manifold.cpu()
                    tree = construct_tree(torch.tensor([i for i in range(data['num_nodes'])]).long(),
                                          manifold,
                                          model.embeddings, model.ass_mat, height=self.configs.height,
                                          num_nodes=embeddings.shape[0])
                    tree_graph = to_networkx_tree(tree, manifold, height=self.configs.height)
                    predicts = decoding_cluster_from_tree(manifold, tree_graph,
                                                          data['num_classes'], data['num_nodes'],
                                                          height=self.configs.height)
                    trues = data['labels']
                    acc, nmi, f1, ari = [], [], [], []
                    for step in range(n_cluster_trials):
                        metrics = cluster_metrics(trues, predicts)
                        acc_, nmi_, f1_, ari_ = metrics.evaluateFromLabel(use_acc=False)
                        acc.append(acc_)
                        nmi.append(nmi_)
                        f1.append(f1_)
                        ari.append(ari_)
                    acc, nmi, f1, ari = np.mean(acc), np.mean(
                        nmi), np.mean(f1), np.mean(ari)
                    if acc > best_cluster['acc']:
                        best_cluster['acc'] = acc
                        best_cluster_result['acc'] = [acc, nmi, f1, ari]
                        torch.save(model, "model.pt")
                    if nmi > best_cluster['nmi']:
                        best_cluster['nmi'] = nmi
                        best_cluster_result['nmi'] = [acc, nmi, f1, ari]
                        logger.info('------------------Saving best model-------------------')
                        torch.save(model.state_dict(), f"./checkpoints/{self.configs.save_path}")
                    if f1 > best_cluster['f1']:
                        best_cluster['f1'] = f1
                        best_cluster_result['f1'] = [acc, nmi, f1, ari]
                    if ari > best_cluster['ari']:
                        best_cluster['ari'] = ari
                        best_cluster_result['ari'] = [acc, nmi, f1, ari]
                    logger.info(
                        f"Epoch {epoch}: ACC: {acc}, NMI: {nmi}, F1: {f1}, ARI: {ari}")
                    logger.info(
                        "-------------------------------------------------------------------------")
            logger.info('------------------Loading best model-------------------')
            model.load_state_dict(torch.load(f"./checkpoints/{self.configs.save_path}"))
            model.eval()
            embeddings = model(data, device).detach().cpu()
            manifold = model.manifold.cpu()
            tree = construct_tree(torch.tensor([i for i in range(data['num_nodes'])]).long(),
                                  manifold,
                                  model.embeddings, model.ass_mat, height=self.configs.height,
                                  num_nodes=embeddings.shape[0])
            tree_graph = to_networkx_tree(tree, manifold, height=self.configs.height)
            _, color_dict = plot_leaves(tree_graph, manifold, embeddings, data['labels'], height=self.configs.height,
                        save_path=f"./results/{self.configs.dataset}/{self.configs.dataset}_hyp_h{self.configs.height}_{exp_iter}_true.pdf")
            # plot_nx_graph(tree_graph, root=data['num_nodes'],
            #               save_path=f"./results/{self.configs.dataset}/{self.configs.dataset}_hyp_h{self.configs.height}_{exp_iter}_nx.pdf")
            # predicts = decoding_cluster_from_tree(manifold, tree_graph,
            #                                       data['num_classes'], data['num_nodes'],
            #                                       height=self.configs.height)
            # trues = data['labels']
            # metrics = cluster_metrics(trues, predicts)
            # metrics.clusterAcc()
            # new_pred = metrics.new_predicts
            # plot_leaves(tree_graph, manifold, embeddings, new_pred, height=self.configs.height,
            #                             save_path=f"./results/{self.configs.dataset}/{self.configs.dataset}_hyp_h{self.configs.height}_{exp_iter}_pred.pdf",
            #             colors_dict=color_dict)
            total_nmi.append(best_cluster['nmi'])
            total_ari.append(best_cluster['ari'])
            for k, result in best_cluster_result.items():
                acc, nmi, f1, ari = result
                logger.info(
                    f"Best Results according to {k}: ACC: {acc}, NMI: {nmi}, F1: {f1}, ARI: {ari} \n")
        logger.info(f"NMI: {np.mean(total_nmi)}+-{np.std(total_nmi)}, "
                    f"ARI: {np.mean(total_ari)}+-{np.std(total_ari)}")