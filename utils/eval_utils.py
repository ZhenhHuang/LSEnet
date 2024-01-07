import numpy as np
from sklearn import metrics
from munkres import Munkres


class cluster_metrics:
    def __init__(self, trues, predicts):
        self.trues = trues
        self.predicts = predicts

    def clusterAcc(self):
        l1 = list(set(self.trues))
        l2 = list(set(self.predicts))
        num1 = len(l1)
        num2 = len(l2)
        ind = 0
        if num1 != num2:
            for i in l1:
                if i in l2:
                    pass
                else:
                    self.predicts[ind] = i
                    ind += 1
        l2 = list(set(self.predicts))
        num2 = len(l2)
        if num1 != num2:
            raise "class number not equal!"
        """compute the cost of allocating c1 in L1 to c2 in L2"""
        cost = np.zeros((num1, num2), dtype=int)
        for i, c1 in enumerate(l1):
            maps = [i1 for i1, e1 in enumerate(self.predicts) if e1 == c1]
            for j, c2 in enumerate(l2):
                maps_d = [i1 for i1 in maps if self.predicts[i1] == c2]
                cost[i, j] = len(maps_d)

        mks = Munkres()
        cost = cost.__neg__().tolist()
        index = mks.compute(cost)
        new_predicts = np.zeros(len(self.predicts))
        for i, c in enumerate(l1):
            c2 = l2[index[i][1]]
            allocate_index = [ind for ind, elm in enumerate(self.predicts) if elm == c2]
            new_predicts[allocate_index] = c
        self.new_predicts = new_predicts
        acc = metrics.accuracy_score(self.trues, new_predicts)
        f1_macro = metrics.f1_score(self.trues, new_predicts, average='macro')
        precision_macro = metrics.precision_score(
            self.trues, new_predicts, average='macro')
        recall_macro = metrics.recall_score(
            self.trues, new_predicts, average='macro')
        f1_micro = metrics.f1_score(self.trues, new_predicts, average='micro')
        precision_micro = metrics.precision_score(
            self.trues, new_predicts, average='micro')
        recall_micro = metrics.recall_score(
            self.trues, new_predicts, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluateFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.trues, self.predicts)
        adjscore = metrics.adjusted_rand_score(self.trues, self.predicts)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusterAcc()
        return acc, nmi, f1_macro, adjscore