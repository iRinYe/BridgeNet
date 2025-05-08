"""
@File    : main.py
@Time    : 15/1/25 12:28
@Author  : iRinYe
@Contact : iRinYeh@outlook.com
"""

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as tg_DataLoader
from lib.lib import ProteinData_Torch, ModelEvaluator, DeepLearningToolkit
from model import tAMPerNet, DeepFRINet, CoEnzymeNet


def tAMPer():
    test_dataset = ProteinData_Torch('tmp/tAMPer/tAMPer_test.pkl')
    model = tAMPerNet(2)
    model.load_state_dict(torch.load(f"models/tAMPer.pt"))
    test_loader = tg_DataLoader(test_dataset, 128, False)
    trainer = DeepLearningToolkit()
    trainer.tAMPer_test(model, test_loader)


def DeepFRI(task):
    if task == 'EC':
        test_dataset = ProteinData_Torch(f'tmp/DeepFRI_EC/DeepFRI_EC_test_level4.pkl')
    elif task == 'BP':
        test_dataset = ProteinData_Torch(f'tmp/DeepFRI_GO/DeepFRI_GO_test_biological_process.pkl')
    elif task == 'MF':
        test_dataset = ProteinData_Torch(f'tmp/DeepFRI_GO/DeepFRI_GO_test_molecular_function.pkl')
    elif task == 'CC':
        test_dataset = ProteinData_Torch(f'tmp/DeepFRI_GO/DeepFRI_GO_test_cellular_component.pkl')

    model = DeepFRINet(len(test_dataset.label_list[0]))
    model.load_state_dict(torch.load(f"models/DeepFRI_{task}.pt"))
    test_loader = tg_DataLoader(test_dataset, 128, False)
    trainer = DeepLearningToolkit()
    trainer.DeepFRI_test(model, test_loader)


def CoEnzyme():
    CoEnzyme_dataset = ProteinData_Torch('tmp/CoEnzyme/CoEnzyme.pkl')

    ModelEvaluator.seed_everything()
    train_size = int(0.7 * len(CoEnzyme_dataset))
    val_size = int(0.1 * len(CoEnzyme_dataset))
    test_size = len(CoEnzyme_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(CoEnzyme_dataset, [train_size, val_size, test_size])
    model = CoEnzymeNet()
    model.load_state_dict(torch.load(f"models/CoEnzyme.pt"))
    test_loader = tg_DataLoader(test_dataset, 128, False)
    trainer = DeepLearningToolkit()
    trainer.CoEnzyme_test(model, test_loader)


tAMPer()
DeepFRI(task='EC')
DeepFRI(task='BP')
DeepFRI(task='MF')
DeepFRI(task='CC')
CoEnzyme()
