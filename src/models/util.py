from torchmetrics import F1Score, Accuracy


def get_classifiation_metric(dataset):
    d = {"ppi": "f1", "cora": "acc"}
    mapping = {"f1": F1Score, "acc": Accuracy}

    metric = d[dataset]
    return mapping[metric], metric
