import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from hyperSE import HyperSE
from geoopt.optim import RiemannianAdam
from eval_utils import cluster_metrics
from plot_utils import plot_leaves
from decode import construct_tree
from dataset import load_data
from train_utils import EarlyStopping
from logger import create_logger


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        data = load_data(self.configs)

        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")
            model = HyperSE(num_nodes=data['num_nodes'], height=self.configs.height).to(device)
            optimizer = RiemannianAdam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)

            early_stopping = EarlyStopping(self.configs.patience)
            logger.info("--------------------------Training Start-------------------------")
            best_cluster = {'acc': 0, 'nmi': 0, 'f1': 0, 'ari': 0}
            best_cluster_result = {}
            n_cluster_trials = self.configs.n_cluster_trials

            for epoch in range(1, self.configs.epochs + 1):
                model.train()
                loss = model.loss(data['edge_index'], data['degrees'], data['weight'], device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.info(f"Epoch {epoch}: train_loss={loss.item()}")
                early_stopping(loss, model, self.configs.save_path)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

            embeddings = model().detach().cpu()
            plot_leaves(embeddings.numpy(), data['labels'], height=self.configs.height)
            tree = construct_tree([i for i in range(data['num_nodes'])],
                                  embeddings, K=self.configs.height,
                                  c=0.999/(self.configs.height + 1), k=1)
            loss = model.loss(data['edge_index'], data['degrees'], data['weight'], device)
            
            #     if epoch % self.configs.eval_freq == 0:
            #         logger.info("---------------Evaluation Start-----------------")
            model.eval()
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=len(np.unique(data['labels'])))
            predicts = kmeans.fit_predict(embeddings)
            trues = data['labels']

            acc, nmi, f1, ari = [], [], [], []
            for step in range(n_cluster_trials):
                metrics = cluster_metrics(trues, predicts)
                acc_, nmi_, f1_, ari_ = metrics.evaluateFromLabel()
                acc.append(acc_)
                nmi.append(nmi_)
                f1.append(f1_)
                ari.append(ari_)
            acc, nmi, f1, ari = np.mean(acc), np.mean(
                nmi), np.mean(f1), np.mean(ari)
            #         if acc > best_cluster['acc']:
            #             best_cluster['acc'] = acc
            #             best_cluster_result['acc'] = [acc, nmi, f1, ari]
            #             torch.save(model, "model.pt")
            #         if nmi > best_cluster['nmi']:
            #             best_cluster['nmi'] = nmi
            #             best_cluster_result['nmi'] = [acc, nmi, f1, ari]
            #         if f1 > best_cluster['f1']:
            #             best_cluster['f1'] = f1
            #             best_cluster_result['f1'] = [acc, nmi, f1, ari]
            #         if ari > best_cluster['ari']:
            #             best_cluster['ari'] = ari
            #             best_cluster_result['ari'] = [acc, nmi, f1, ari]
            logger.info(
                f"Epoch {epoch}: ACC: {acc}, NMI: {nmi}, F1: {f1}, ARI: {ari}")
            logger.info(
                "-------------------------------------------------------------------------")
            # for k, result in best_cluster_result.items():
            #     acc, nmi, f1, ari = result
            #     logger.info(
            #         f"Best Results according to {k}: ACC: {acc}, NMI: {nmi}, F1: {f1}, ARI: {ari} \n")
