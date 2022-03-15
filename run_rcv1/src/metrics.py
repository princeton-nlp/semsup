"""
Metrics for multi-label text classification
"""

import numpy as np
from scipy.special import expit
import itertools
import copy

# Metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_score,
    recall_score,
    label_ranking_average_precision_score,
    coverage_error
)

def get_ancestors(data_args, label2id):
    """
    Get the ancestors of all the nodes using the hierarchy file given.
    """

    if data_args.task_name == 'rcv1':
        # Open the file and get all the lines
        lines = open(data_args.hierarchy_file, 'r').readlines()

        # Store the parents for each node as a list
        # Store all the lists in a dict
        parents = {}

        # Go over all the lines in the file
        label2id_copy = copy.deepcopy(label2id)
        label2id_copy['Root'] = -1

        for line in lines:
            split_line = line.strip().split()
            if split_line[1] != "None" and split_line[3] in label2id_copy.keys():
                parents[split_line[3]] = split_line[1]

        def get_ancestors_recursion(node):
            if node == 'Root':
                return ['Root']
            else:
                return [node] + get_ancestors_recursion(parents[node])

        # Collect all the ancestors
        ancestors = {}
        for key in parents.keys():
            ancestors[key] = get_ancestors_recursion(key)
        
        # Convert labels to IDS
        for key in ancestors.keys():
            ancestors[key] = [label2id_copy[i] for i in ancestors[key]]
        
        return ancestors

    else:
        raise("Hierarchal metrics support only for RCV1.")


def compute_hierarchical_micro_f1(preds, label_ids, ancestor_dict, id2label):
    """
    Compute the hierarchical micro F-1.
    """
    precision = [0, 0]
    recall = [0, 0]
    
    # Loop over all the instances
    for i in range(preds.shape[0]):
        true_nodes = []
        predicted_nodes = []
        
        # Collect true node ancestors
        for j in range(preds.shape[1]):
            if label_ids[i][j] == 1:
                true_nodes = true_nodes + ancestor_dict[id2label[j]]
            if preds[i][j] == 1:
                predicted_nodes = predicted_nodes + ancestor_dict[id2label[j]]
        
        # Compute the intersection
        # Numerator
        precision[0] += len(set(true_nodes) & set(predicted_nodes))
        recall[0] += len(set(true_nodes) & set(predicted_nodes))
        
        # Denominator
        precision[1] += len(set(predicted_nodes))
        recall[1] += len(set(true_nodes))

    # Compute precision and recall
    # Handle zero-division errors
    if precision[1] == 0:
        h_precision = 1
    else:
        h_precision = precision[0] / precision[1]

    if recall[1] == 0:
        h_recall = 1
    else:
        h_recall = recall[0] / recall[1]

    if h_precision == 0 and h_recall == 0:
        h_f1 = 0
    else:
        h_f1 = 2 * h_precision * h_recall / (h_precision + h_recall)

    return h_f1


def multilabel_metrics(data_args, id2label, label2id, fbr):
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

        # METRIC 1: Compute accuracy
        if 'accuracy' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                accuracy_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = np.sum(p.label_ids == accuracy_preds) / accuracy_preds.size * 100
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['accuracy'] = best_threshold
            accuracy = performance[best_threshold]
        else:
            accuracy_preds = np.where(preds > fbr['accuracy'], 1, 0)
            accuracy = np.sum(p.label_ids == accuracy_preds) / accuracy_preds.size * 100

        # METRIC 2: Compute the subset accuracy
        if 'subset_accuracy' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                subset_accuracy_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = accuracy_score(p.label_ids, subset_accuracy_preds)
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['subset_accuracy'] = best_threshold
            subset_accuracy = performance[best_threshold]
        else:
            subset_accuracy_preds = np.where(preds > fbr['subset_accuracy'], 1, 0)
            subset_accuracy = accuracy_score(p.label_ids, subset_accuracy_preds)

        # METRIC 3: Macro F-1
        if 'macro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                macro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = f1_score(p.label_ids, macro_f1_preds, average='macro')
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['macro_f1'] = best_threshold
            macro_f1 = performance[best_threshold]
        else:
            macro_f1_preds = np.where(preds > fbr['macro_f1'], 1, 0)
            macro_f1 = f1_score(p.label_ids, macro_f1_preds, average='macro')

        # METRIC 4: Micro F-1
        if 'micro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                micro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = f1_score(p.label_ids, micro_f1_preds, average='micro')
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['micro_f1'] = best_threshold
            micro_f1 = performance[best_threshold]
        else:
            micro_f1_preds = np.where(preds > fbr['micro_f1'], 1, 0)
            micro_f1 = f1_score(p.label_ids, micro_f1_preds, average='micro')

        # METRIC 5: Hierarchical micro F-1
        ancestor_dict = get_ancestors(data_args, label2id)
        if 'hier_micro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                hier_micro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = compute_hierarchical_micro_f1(hier_micro_f1_preds, p.label_ids, ancestor_dict, id2label)
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['hier_micro_f1'] = best_threshold
            hier_micro_f1 = performance[best_threshold]
        else:
            hier_micro_f1_preds = np.where(preds > fbr['hier_micro_f1'], 1, 0)
            hier_micro_f1 = compute_hierarchical_micro_f1(hier_micro_f1_preds, p.label_ids, ancestor_dict, id2label)

        # Multi-label classification report
        # Optimized for Micro F-1
        report = classification_report(p.label_ids, micro_f1_preds, target_names=[id2label[i] for i in range(len(id2label))])
        print(report)

        return {
            "accuracy": accuracy,
            "subset_accuracy": subset_accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "hier_micro_f1": hier_micro_f1,
            "fbr": fbr
        }

    return compute_metrics


def multilabel_label_descriptions_metrics(data_args, id2label, label2id, label_list_dict, fbr):
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

        # Determine if it's validation or prediction
        is_validation = False if fbr else True

        # Define the range over which the best fbr value is chosen
        fbr_low = 0.0
        fbr_high = 1.0
        fbr_step = 0.05

        # METRIC 1: Compute accuracy
        if 'accuracy' not in fbr.keys():
            performance = {}
            for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                accuracy_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = np.sum(p.label_ids == accuracy_preds) / accuracy_preds.size * 100
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['accuracy'] = best_threshold
            accuracy = performance[best_threshold]
        else:
            accuracy_preds = np.where(preds > fbr['accuracy'], 1, 0)
            accuracy = np.sum(p.label_ids == accuracy_preds) / accuracy_preds.size * 100

        # METRIC 2: Compute the subset accuracy
        if 'subset_accuracy' not in fbr.keys():
            performance = {}
            for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                subset_accuracy_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = accuracy_score(p.label_ids, subset_accuracy_preds)
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['subset_accuracy'] = best_threshold
            subset_accuracy = performance[best_threshold]
        else:
            subset_accuracy_preds = np.where(preds > fbr['subset_accuracy'], 1, 0)
            subset_accuracy = accuracy_score(p.label_ids, subset_accuracy_preds)

        # METRIC 3: Macro F-1
        if 'macro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                macro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = f1_score(p.label_ids, macro_f1_preds, average='macro')
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['macro_f1'] = best_threshold
            macro_f1 = performance[best_threshold]
        else:
            macro_f1_preds = np.where(preds > fbr['macro_f1'], 1, 0)
            macro_f1 = f1_score(p.label_ids, macro_f1_preds, average='macro')

        # METRIC 4: Micro F-1
        if 'micro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                micro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = f1_score(p.label_ids, micro_f1_preds, average='micro')
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['micro_f1'] = best_threshold
            micro_f1 = performance[best_threshold]
        else:
            micro_f1_preds = np.where(preds > fbr['micro_f1'], 1, 0)
            micro_f1 = f1_score(p.label_ids, micro_f1_preds, average='micro')

        # METRIC 5: Hierarchical micro F-1
        ancestor_dict = get_ancestors(data_args, label2id)
        if 'hier_micro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                hier_micro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = compute_hierarchical_micro_f1(hier_micro_f1_preds, p.label_ids, ancestor_dict, id2label)
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['hier_micro_f1'] = best_threshold
            hier_micro_f1 = performance[best_threshold]
        else:
            hier_micro_f1_preds = np.where(preds > fbr['hier_micro_f1'], 1, 0)
            hier_micro_f1 = compute_hierarchical_micro_f1(hier_micro_f1_preds, p.label_ids, ancestor_dict, id2label)

        # METRIC X: DELETE ##############################
        if 'm14_micro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(fbr_low, fbr_high, fbr_step * 0.1):
                m14_micro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = f1_score(p.label_ids[:,label2id['M14']], m14_micro_f1_preds[:,label2id['M14']], average='binary')
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['m14_micro_f1'] = best_threshold
            M14_score = performance[best_threshold]
            print("M14 score is: {}".format(M14_score))
            m14_micro_f1_preds = np.where(preds > fbr['m14_micro_f1'], 1, 0)
            print("M14 precision is: {}".format(precision_score(p.label_ids[:,label2id['M14']], m14_micro_f1_preds[:,label2id['M14']], average='binary')))
            print("M14 recall is: {}".format(recall_score(p.label_ids[:,label2id['M14']], m14_micro_f1_preds[:,label2id['M14']], average='binary')))
        else:
            m14_micro_f1_preds = np.where(preds > fbr['m14_micro_f1'], 1, 0)
            M14_score = f1_score(p.label_ids[:,label2id['M14']], m14_micro_f1_preds[:,label2id['M14']], average='binary')
            print("M14 score is: {}".format(M14_score))
            print("M14 precision is: {}".format(precision_score(p.label_ids[:,label2id['M14']], m14_micro_f1_preds[:,label2id['M14']], average='binary')))
            print("M14 recall is: {}".format(recall_score(p.label_ids[:,label2id['M14']], m14_micro_f1_preds[:,label2id['M14']], average='binary')))
        #################################################

        # Multi-label classification report
        # Optimized for Micro F-1 (the use of micro_f1_preds)
        print("*** Classification report for all the classes ***")
        micro_f1_preds = np.where(preds > fbr['micro_f1'], 1, 0)
        report = classification_report(p.label_ids, micro_f1_preds, target_names=[id2label[i] for i in p.represented_labels])
        print(report)
        print("********************************************")

        # Classification report only for classes that appeared in the validation/prediction set but not the train set
        # Get the classes which belong to the validation set but not the train set
        if data_args.evaluation_type == 'gzs':
            key = 'validation' if is_validation else 'test'
            set_difference = list(set(label_list_dict[key]).difference(set(label_list_dict['train'])))
            set_difference = [label2id[label] for label in set_difference]
            # Take an intersection with the labels that are represented in the current data
            set_difference = list(set(set_difference).intersection(set(p.represented_labels)))
            set_difference.sort()

            if len(set_difference) > 0:
                print("*** Classification report for classes not in the train set ***")
                target_names = [id2label[i] for i in set_difference]

                # HACK: Find macro F-1 optimized for the unseen labels
                performance = {}
                for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                    unseen_macro_f1_preds = np.where(preds[:,set_difference] > threshold, 1, 0)
                    performance[threshold] = f1_score(p.label_ids[:,set_difference], unseen_macro_f1_preds, average='macro')
                # Choose the best threshold
                best_threshold = max(performance, key=performance.get)
                unseen_macro_f1 = performance[best_threshold]

                report = classification_report(p.label_ids[:,set_difference], np.where(preds[:,set_difference] > best_threshold, 1, 0), target_names=target_names)
                print(report)

                print("Best threshold for unseen macro F-1: {}".format(best_threshold))
                print("********************************************")

        # Print the thresholds used
        print("********************************************")
        print("Thresholds used: {}".format(fbr))
        print("********************************************")

        return {
            "accuracy": accuracy,
            "subset_accuracy": subset_accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "hier_micro_f1": hier_micro_f1,
            "unseen_micro_f1": unseen_macro_f1,
            "fbr": fbr,
            "unseen_average_prediction_score": np.average(preds[:,set_difference]),
            "seen_average_prediction_score": np.average(preds[:, [label2id[label] for label in label_list_dict['train']]]),
        }

    return compute_metrics


def multilabel_label_descriptions_per_class_threshold_metrics(data_args, id2label, label2id, label_list_dict, fbr):
    """
    Metrics function used for multilabel classification.
    Choose a different threshold for each class.

    Don't compute accuracy and subset accuracy.

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

        # Determine if it's validation or prediction
        is_validation = False if fbr else True

        # Define the range over which the best fbr value is chosen
        fbr_low = 0.0
        fbr_high = 1.0
        fbr_step = 0.05

        # METRIC 3: Macro F-1
        if 'macro_f1' not in fbr.keys():
            # Store the best threshold for each class
            best_thresholds = []
            # Loop over all the classes
            for i in range(preds.shape[1]):
                performance = {}
                for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                    macro_f1_preds = np.where(preds[:,i] > threshold, 1, 0)
                    performance[threshold] = f1_score(p.label_ids[:,i], macro_f1_preds, average='binary')
                # Choose the best threshold
                best_thresholds.append(max(performance, key=performance.get))
            fbr['macro_f1'] = np.array(best_thresholds)
            macro_f1 = f1_score(p.label_ids, np.where(preds > fbr['macro_f1'], 1, 0), average='macro')
        else:
            macro_f1 = f1_score(p.label_ids, np.where(preds > fbr['macro_f1'], 1, 0), average='macro')

        # NOTE: The following function was only for debugging purposes
        # METRIC: Macro F-1 with global threshold
        if 'global_macro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                global_macro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = f1_score(p.label_ids, global_macro_f1_preds, average='macro')
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['global_macro_f1'] = best_threshold
            global_macro_f1 = performance[best_threshold]
        else:
            global_macro_f1_preds = np.where(preds > fbr['global_macro_f1'], 1, 0)
            global_macro_f1 = f1_score(p.label_ids, global_macro_f1_preds, average='macro')
        
        # METRIC: LRAP (Label ranking average precision)
        total_lrap = label_ranking_average_precision_score(p.label_ids, preds)

        # Multi-label classification report
        # Optimized for Micro F-1 (the use of micro_f1_preds)
        print("*** Classification report for all the classes ***")
        macro_f1_preds = np.where(preds > fbr['macro_f1'], 1, 0)
        report = classification_report(p.label_ids, macro_f1_preds, target_names=[id2label[i] for i in p.represented_labels])
        print(report)
        print("********************************************")

        # Classification report only for classes that appeared in the validation/prediction set but not the train set
        # Get the classes which belong to the validation set but not the train set
        unseen_macro_f1 = 0.
        if data_args.evaluation_type == 'gzs':
            key = 'validation' if is_validation else 'test'
            set_difference = list(set(label_list_dict[key]).difference(set(label_list_dict['train'])))
            set_difference = [label2id[label] for label in set_difference]
            # Take an intersection with the labels that are represented in the current data
            set_difference = list(set(set_difference).intersection(set(p.represented_labels)))
            set_difference.sort()

            if len(set_difference) > 0:
                
                # METRIC: Unseen Macro F-1
                # Store the best threshold for each class
                best_thresholds = []
                # Loop over all the classes
                for i in set_difference:
                    performance = {}
                    for threshold in np.arange(fbr_low, fbr_high, fbr_step * 0.1):
                        unseen_micro_f1_preds = np.where(preds[:,i] > threshold, 1, 0)
                        performance[threshold] = f1_score(p.label_ids[:,i], unseen_micro_f1_preds, average='binary')
                    # Choose the best threshold
                    best_thresholds.append(max(performance, key=performance.get))
                fbr['unseen_macro_f1'] = np.array(best_thresholds)
                unseen_macro_f1 = f1_score(p.label_ids[:,set_difference], np.where(preds[:,set_difference] > fbr['unseen_macro_f1'], 1, 0), average='macro')

                # METRIC: Unseen Macro F-1 with a global threshold
                performance = {}
                for threshold in np.arange(fbr_low, fbr_high, fbr_step):
                    global_unseen_macro_f1_preds = np.where(preds[:,set_difference] > threshold, 1, 0)
                    performance[threshold] = f1_score(p.label_ids[:,set_difference], global_unseen_macro_f1_preds, average='macro')
                # Choose the best threshold
                best_threshold = max(performance, key=performance.get)
                global_unseen_macro_f1 = performance[best_threshold]
                fbr['global_unseen_macro_f1'] = best_threshold

                # METRIC: Unseen LRAP
                unseen_lrap = label_ranking_average_precision_score(p.label_ids[:,set_difference], preds[:,set_difference])

                # Print the classification report
                print("*** Classification report for classes not in the train set ***")
                target_names = [id2label[i] for i in set_difference]
                report = classification_report(p.label_ids[:,set_difference], np.where(preds[:,set_difference] > fbr['unseen_macro_f1'], 1, 0), target_names=target_names)
                print(report)

                print("Best thresholds for unseen micro F-1: {}".format(best_thresholds))
                print("********************************************")

        return {
            "macro_f1": macro_f1,
            "global_macro_f1": global_macro_f1,
            "unseen_macro_f1": unseen_macro_f1,
            "global_unseen_macro_f1": global_unseen_macro_f1,
            "total_lrap": total_lrap,
            "unseen_lrap": unseen_lrap,
            "fbr": fbr,
            "unseen_average_prediction_score": np.average(preds[:,set_difference]),
            "seen_average_prediction_score": np.average(preds[:, [label2id[label] for label in label_list_dict['train']]]),
        }

    return compute_metrics


def multilabel_label_descriptions_ranking_metrics(data_args, id2label, label2id, label_list_dict, fbr):
    """
    Ranking metrics for multilabel classification.
    LRAP and coverage error.
    Higher is better for LRAP and lower is better for coverage error.

    fbr is a flag which tells us if it is train or validation

    Datasets: RCV1-V2
    """
    def compute_metrics(p):
        # Collect the logits
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        # Compute the logistic sigmoid
        preds = expit(preds)

        # Determine if it's validation or prediction
        is_validation = False if fbr else True

        # Compute the metrics over all the labels

        # Choose only examples which have at least one positive in train_labels
        bool_arr = np.sum(p.label_ids, axis=1) > 0        
        
        # METRIC: LRAP (Label ranking average precision)
        total_lrap = label_ranking_average_precision_score(p.label_ids[bool_arr], preds[bool_arr])

        # METRIC: Coverage error
        total_coverage_error = coverage_error(p.label_ids[bool_arr], preds[bool_arr])

        # Compute the metrics for seen and unseen classes
        seen_lrap, seen_coverage_error, unseen_lrap, unseen_coverage_error = 0., 0., 0., 0.
        if data_args.evaluation_type != 'seen':
            key = 'validation' if is_validation else 'test'
            set_difference = list(set(label_list_dict[key]).difference(set(label_list_dict['train'])))
            set_difference = [label2id[label] for label in set_difference]
            # Take an intersection with the labels that are represented in the current data
            validation_labels = list(set(set_difference).intersection(set(p.represented_labels)))
            validation_labels.sort()

            train_labels = [label2id[label] for label in label_list_dict['train']]

            # Compute metrics only on the train labels
            # Choose only examples which have at least one positive in train_labels
            bool_arr = np.sum(p.label_ids[:,train_labels], axis=1) > 0

            # METRIC: LRAP (Label ranking average precision)
            seen_lrap = label_ranking_average_precision_score(p.label_ids[bool_arr][:,train_labels], preds[bool_arr][:,train_labels])

            # METRIC: Coverage error
            seen_coverage_error = coverage_error(p.label_ids[bool_arr][:,train_labels], preds[bool_arr][:,train_labels])

            # Copmute metrics only on the eval labels            
            if len(validation_labels) > 0:

                # Choose only examples which have at least one positive in validation_labels
                bool_arr = np.sum(p.label_ids[:,validation_labels], axis=1) > 0
                
                # METRIC: LRAP (Label ranking average precision)
                unseen_lrap = label_ranking_average_precision_score(p.label_ids[bool_arr][:,validation_labels], preds[bool_arr][:,validation_labels])

                # METRIC: Coverage error
                # BUG: The following line does not work for some reason
                # unseen_coverage_error = coverage_error(p.label_ids[bool_arr][:,validation_labels], preds[bool_arr][:,validation_labels])

        return {
            "total_lrap": total_lrap,
            "total_coverage_error": total_coverage_error,
            "seen_lrap": seen_lrap,
            "seen_coverage_error": seen_coverage_error,
            "unseen_lrap": unseen_lrap,
            # "unseen_coverage_error": unseen_coverage_error,
            "fbr": {"total_lrap": total_lrap},
        }

    return compute_metrics