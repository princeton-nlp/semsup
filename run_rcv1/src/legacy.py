######################################################################################################
"""
Old function from metrics.py
1/2/2022
"""

def multilabel_metrics(data_args, id2label, label2id, fbr=None):
    """
    Metrics function used for multilabel classification.
    Datasets: RCV1-V2

    :fbr : A dict containing global thresholds to be used for selecting a class.
    We use global thresholds because we want to handle unseen classes,
    for which the threshold is not known in advance.
    """
    def compute_metrics(p):
        # Collect the logits
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        # Compute the logistic sigmoid
        preds = expit(preds)

        # Convert them to 0 and 1's based on prediction
        threshold = 0.5
        preds = np.where(preds > threshold, 1, 0)

        # Compute the subset accuracy
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        subset_accuracy = accuracy_score(p.label_ids, preds)

        # Compute the standard accuracy
        accuracy = np.sum(p.label_ids == preds) / preds.size * 100

        # Multi-label F-1
        macro_f1 = f1_score(p.label_ids, preds, average='macro')
        micro_f1 = f1_score(p.label_ids, preds, average='micro')

        # Hierarchical micro F1
        ancestor_dict = get_ancestors(data_args, label2id)
        hier_micro_f1 = compute_hierarchical_micro_f1(preds, p.label_ids, ancestor_dict, id2label)        

        # Multi-label classification report
        report = classification_report(p.label_ids, preds, target_names=[id2label[i] for i in range(len(id2label))])
        print(report)

        return {
            "accuracy": accuracy,
            "subset_accuracy": subset_accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "hier_micro_f1": hier_micro_f1,
        }

    return compute_metrics

######################################################################################################    