import os
from lib.GraphBuilder import pdb_to_graph
os.environ['NUMEXPR_MAX_THREADS'] = '16'
from torch_geometric.data import Data
from torch.utils.data import  Dataset
import torch.nn.functional as F
import pickle
import random
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, \
    average_precision_score, matthews_corrcoef
from torch import nn
from tqdm import tqdm


class ProteinData_Torch(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.encoded_sequence_list = data.encoded_sequence_list
        self.graph_data_list = data.graph_data_list
        self.sequence_list = data.sequence_list
        self.label_list = data.label_list
        if all(graph is None for graph in self.graph_data_list):
            self.graph_data_list = [torch.tensor([]) for _ in self.graph_data_list]
        if all(label is None for label in self.label_list):
            self.label_list = [torch.tensor([]) for _ in self.label_list]
    def __len__(self):
        return len(self.label_list)
    def __getitem__(self, idx):
        data = {
            'graph_data': self.graph_data_list[idx],
            'encoded_sequence': self.encoded_sequence_list[idx],
            'sequence': self.sequence_list[idx],
            'label': self.label_list[idx],
        }
        if hasattr(self, 'seq_embedding'):
            data['seq_embedding'] = self.seq_embedding[idx]
        return data


class ProteinData:
    def __init__(self):
        self.graph_data_list = []
        self.encoded_sequence_list = []
        self.sequence_list = []
        self.label_list = []
    def add_entry(self, graph_data, encoded_sequence, sequence, label=None):
        self.graph_data_list.append(graph_data)
        self.encoded_sequence_list.append(encoded_sequence)
        self.sequence_list.append(sequence)
        self.label_list.append(label)
    def __len__(self):
        return len(self.encoded_sequence_list)
    def __getitem__(self, idx):
        return {
            "graph_data": self.graph_data_list[idx],
            "encoded_sequence": self.encoded_sequence_list[idx],
            "sequence": self.sequence_list[idx],
            "label": self.label_list[idx] if self.label_list else None
        }


class ProteinProcessor:
    def __init__(self, max_length, padding_coord=(-99, -99, -99), padding_aa='X'):
        self.max_length = max_length
        self.padding_coord = padding_coord
        self.padding_aa = padding_aa
    def process_pdb_files(self, protein_data: ProteinData, pdb_path):
        pdb_files = [f for f in os.listdir(pdb_path) if f.endswith(".pdb")]
        for filename in tqdm(pdb_files):
            file_path = os.path.join(pdb_path, filename)
            graph, edge_index, sequence = pdb_to_graph(file_path, self.max_length, self.padding_aa, self.padding_coord)
            if graph is not None:
                encoded_sequence = self.encode_sequence(sequence)
                graph = Data(x=graph, edge_index=edge_index)
                protein_data.add_entry(graph, encoded_sequence, sequence)
    def process_seqs(self, protein_data: ProteinData, seq_list):
        for record in tqdm(seq_list):
            sequence = record[0]
            label = record[1]
            num_fill = self.max_length - len(sequence)
            if num_fill > 0:
                sequence = sequence + (self.padding_aa * num_fill)
            elif num_fill < 0:
                sequence = sequence[:self.max_length]
            encoded_sequence = self.encode_sequence(sequence)
            protein_data.add_entry(None, encoded_sequence, sequence, label)
    def encode_sequence(self, sequence):
        encoded_array = np.zeros((len(sequence), 20), dtype=int)
        encoding_dict = {'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
                         'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
                         'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
                         'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
                         'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
                         'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
                         'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
                         'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
                         'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
                         'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
                         'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
                         'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
                         'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
                         'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
                         'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
                         'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
                         'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
                         'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
                         'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
                         'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
                         'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         }
        for idx, amino_acid in enumerate(sequence):
            if amino_acid in encoding_dict:
                encoded_array[idx] = encoding_dict[amino_acid]
            else:
                encoded_array[idx] = [0] * 20
        return torch.tensor(encoded_array, dtype=torch.float) / 10


class ModelEvaluator:
    @staticmethod
    def seed_everything(seed=3407):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    @staticmethod
    def calculate_fmax(y_true, y_scores):
        best_fmax = 0
        best_threshold = 0
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i]) == 0:
                continue
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            max_f1 = np.max(f1_scores)
            threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
            if max_f1 > best_fmax:
                best_fmax = max_f1
                best_threshold = threshold
        return best_fmax, best_threshold
    @staticmethod
    def cal_perf(y_test, y_pred, isMultiLabel=False):
        if isMultiLabel == False:
            y_test = np.hstack(y_test)
            y_pred = np.vstack(y_pred)
            acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))
            auc = roc_auc_score(y_test, y_pred[:, 1])
            precision = metrics.precision_score(y_test, np.argmax(y_pred, axis=1))
            recall = metrics.recall_score(y_test, np.argmax(y_pred, axis=1))
            f1 = metrics.f1_score(y_test, np.argmax(y_pred, axis=1))
            aupr = average_precision_score(y_test, y_pred[:, 1])
            mcc = matthews_corrcoef(y_test, np.argmax(y_pred, axis=1))
            print(
                f"  Accuracy: {round(acc, 3)}; AUC: {round(auc, 3)}; Precision: {round(precision, 3)}; Recall: {round(recall, 3)}; F1 score: {round(f1, 3)}; AUPR: {round(aupr, 3)}; MCC: {round(mcc, 3)}")
        else:
            y_test = np.concatenate(y_test)
            y_pred = np.concatenate(y_pred)
            acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))
            auc = roc_auc_score(y_test, y_pred, multi_class='ovo', average='macro')
            precision = metrics.precision_score(y_test, np.argmax(y_pred, axis=1), average='macro')
            recall = metrics.recall_score(y_test, np.argmax(y_pred, axis=1), average='macro')
            f1 = metrics.f1_score(y_test, np.argmax(y_pred, axis=1), average='macro')
            print(
                f"  Accuracy: {round(acc, 3)}; AUC: {round(auc, 3)}; Precision: {round(precision, 3)}; Recall: {round(recall, 3)}; F1 score: {round(f1, 3)};")
        return {
            'acc': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }


class DeepLearningToolkit:
    @staticmethod
    def tAMPer_test(model, dataloader):
        device = torch.device('cuda:0')
        model.to(device)
        loss_list = []
        label_list = []
        pred_list = []
        test_loss = 0
        len_testset = len(dataloader.dataset)
        loss_func = nn.CrossEntropyLoss()
        with torch.no_grad():
            model.eval()
            for batch_dict in tqdm(dataloader):
                seq = batch_dict['encoded_sequence'].to(device)
                label = batch_dict['label'].to(device)
                pred = model(seq)
                loss = loss_func(pred, label)
                test_loss += (loss.item() * len(seq))
                label_list.append(label.detach().cpu().numpy())
                pred_list.append(pred.detach().cpu().numpy())
        loss_list.append([test_loss / len_testset])
        pref = ModelEvaluator.cal_perf(label_list, pred_list, isMultiLabel=False)
        return pref
    @staticmethod
    def DeepFRI_test(model, dataloader):
        device = torch.device('cuda:0')
        model.to(device)
        label_list = []
        pred_list = []
        with torch.no_grad():
            model.eval()
            for batch_dict in tqdm(dataloader):
                seq = batch_dict['encoded_sequence'].to(device)
                label = batch_dict['label'].to(device)
                pred = model(seq.flatten(1))
                label_list.append(label.detach().cpu().numpy())
                pred_list.append(F.sigmoid(pred).detach().cpu().numpy())
        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)
        fmax, _ = ModelEvaluator.calculate_fmax(label_list, pred_list)
        print(fmax)
        return fmax
    @staticmethod
    def CoEnzyme_test(model, dataloader):
        device = torch.device('cuda:0')
        model.to(device)
        label_list = []
        pred_list = []
        with torch.no_grad():
            model.eval()
            for batch_dict in tqdm(dataloader):
                seq = batch_dict['encoded_sequence'].to(device)
                label = batch_dict['label'].to(device)
                pred = model(seq)
                label_list.append(label.detach().cpu().numpy())
                pred_list.append(F.softmax(pred, dim=1).detach().cpu().numpy())
        pref = ModelEvaluator.cal_perf(label_list, pred_list, isMultiLabel=False)
        return pref