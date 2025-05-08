import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(1024, 1024)
        position = torch.arange(0, 1024, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 1024, 2).float() * (-math.log(10000.0) / 1024))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class Seq_Module(nn.Module):
    def __init__(self):
        super(Seq_Module, self).__init__()
        self.positional_encoding = PositionalEncoding()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=4,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=1024,
            nhead=4,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True
        )
        self.input_projection = nn.Linear(20480, 1024)
        self.output_projection = nn.Linear(1024, 20480)
        self.decoder = nn.TransformerDecoder(decoder_layer, 2)
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        projected_input = self.input_projection(x)
        projected_input = projected_input.unsqueeze(1)
        input_with_pe = self.positional_encoding(projected_input)
        encoded = self.encoder(input_with_pe)
        decoder_input_with_pe = self.positional_encoding(encoded)
        decoded = self.decoder(decoder_input_with_pe, encoded)
        decoded = decoded.squeeze(1)
        reconstructed_output = self.output_projection(decoded)
        return encoded.squeeze(1), reconstructed_output


class tAMPerNet(nn.Module):
    def __init__(self, label_num):
        super(tAMPerNet, self).__init__()
        self.Seq_Module = Seq_Module()
        self.dp = nn.Dropout(0.3)
        self.cls = nn.Sequential(
            nn.Linear(1024, label_num, bias=True),
        )
    def forward(self, encoded_sequence):
        seq_representation, _ = self.Seq_Module(encoded_sequence)
        pred = self.cls(self.dp(seq_representation))
        return pred


class DeepFRINet(nn.Module):
    def __init__(self, label_num):
        super(DeepFRINet, self).__init__()
        self.Seq_Module = Seq_Module()
        self.dp = nn.Dropout(0.3)
        self.cls = nn.Sequential(
            nn.Linear(1024, label_num, bias=True),
        )
    def forward(self, encoded_sequence):
        seq_representation, _ = self.Seq_Module(encoded_sequence)
        pred = self.cls(self.dp(seq_representation))
        return pred


class CoEnzymeNet(nn.Module):
    def __init__(self):
        super(CoEnzymeNet, self).__init__()
        self.Seq_Module = Seq_Module()
        self.dp = nn.Dropout(0.3)
        self.fcn = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
        )
    def forward(self, x):
        seq_representation, _ = self.Seq_Module(x)
        classification_output = self.fcn(seq_representation.squeeze(1))
        return classification_output