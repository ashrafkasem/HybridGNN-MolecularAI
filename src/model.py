"""
Hybrid GNN + Molecular Descriptors Model for IC50 Prediction
Combines graph neural network with traditional molecular descriptors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class HybridGNNModel(nn.Module):
    """
    Hybrid model combining:
    1. Graph Neural Network (for molecular structure)
    2. Molecular Descriptors (for chemical properties)
    """
    
    def __init__(self, node_vocab_size, num_node_features, edge_feature_size,
                 descriptor_size=544,  # 512 (fingerprint) + 32 (descriptors)
                 gnn_hidden_dim=128, gnn_num_layers=3, gnn_dropout=0.2, gnn_num_heads=4,
                 regressor_hidden_dim=104, regressor_num_hidden_layers=3, regressor_dropout=0):
        super().__init__()
        
        # ========== GNN Component (unchanged) ==========
        self.gnn = nn.ModuleDict({
            'node_embedding': Embedding(node_vocab_size + 1, gnn_hidden_dim // 2),
            'lin_node_features': Linear(num_node_features - 1, gnn_hidden_dim // 2),
            'edge_embedding': Embedding(edge_feature_size, gnn_hidden_dim),
            'convs': nn.ModuleList([
                GATConv(gnn_hidden_dim, gnn_hidden_dim // gnn_num_heads, heads=gnn_num_heads, dropout=gnn_dropout)
                for _ in range(gnn_num_layers)
            ]),
            'bns': nn.ModuleList([nn.BatchNorm1d(gnn_hidden_dim) for _ in range(gnn_num_layers)]),
            'dropout': nn.Dropout(gnn_dropout)
        })
        
        # ========== Descriptor Processing Branch ==========
        self.descriptor_processor = nn.Sequential(
            nn.Linear(descriptor_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased from 0.2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased from 0.2
        )
        
        # ========== Fusion Layer ==========
        # Combines GNN output (256 from mean+max pool) + descriptors (128) + N (1)
        fusion_input_dim = gnn_hidden_dim * 2 + 128 + 1  # 256 + 128 + 1 = 385
        
        # +1 for N (molecular size)
        self.regressor = nn.ModuleDict({
            'input_layer': nn.Sequential(
                nn.Linear(fusion_input_dim, regressor_hidden_dim),
                nn.BatchNorm1d(regressor_hidden_dim),
                nn.ReLU(),
                nn.Dropout(regressor_dropout)
            ),
            'hidden_layers': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(regressor_hidden_dim, regressor_hidden_dim),
                    nn.BatchNorm1d(regressor_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(regressor_dropout)
                )
                for _ in range(regressor_num_hidden_layers)
            ]),
            'output_layer': nn.Linear(regressor_hidden_dim, 1)
        })

    def forward(self, x, edge_index, edge_attr, batch, N, descriptors):
        """
        Forward pass combining GNN and descriptors.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch assignment
            N: Molecular size feature
            descriptors: Molecular descriptors (fingerprints + properties)
        """
        # ========== GNN Branch ==========
        embed_indices = x[:, 0].long().squeeze()
        other_node_features = x[:, 1:].float().squeeze()
        embedded_node = self.gnn['node_embedding'](embed_indices)
        processed_other_features = self.gnn['lin_node_features'](other_node_features)
        h = torch.cat([embedded_node, processed_other_features], dim=1)

        edge_attr_indices = edge_attr[:, 0].long().squeeze()
        edge_attr_embedded = self.gnn['edge_embedding'](edge_attr_indices)

        for i in range(len(self.gnn['convs'])):
            h = self.gnn['convs'][i](h, edge_index, edge_attr=edge_attr_embedded)
            h = self.gnn['bns'][i](h)
            h = F.relu(h)
            h = self.gnn['dropout'](h)

        # Global pooling for graph-level representation
        graph_embedding = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        
        # ========== Descriptor Branch ==========
        descriptor_features = self.descriptor_processor(descriptors)
        
        # ========== Fusion ==========
        # Combine GNN output + descriptor features + N
        combined = torch.cat([graph_embedding, descriptor_features, N.unsqueeze(1)], dim=1)
        
        # ========== Regressor ==========
        x_reg = self.regressor['input_layer'](combined)
        for layer in self.regressor['hidden_layers']:
            x_reg = layer(x_reg)
        prediction = self.regressor['output_layer'](x_reg)
        
        return prediction


class GNNOnlyModel(nn.Module):
    """
    Standard GNN model without descriptors (for comparison).
    This is your current model.
    """
    
    def __init__(self, node_vocab_size, num_node_features, edge_feature_size,
                 gnn_hidden_dim=128, gnn_num_layers=3, gnn_dropout=0.2, gnn_num_heads=4,
                 regressor_hidden_dim=104, regressor_num_hidden_layers=3, regressor_dropout=0):
        super().__init__()

        self.gnn = nn.ModuleDict({
            'node_embedding': Embedding(node_vocab_size + 1, gnn_hidden_dim // 2),
            'lin_node_features': Linear(num_node_features - 1, gnn_hidden_dim // 2),
            'edge_embedding': Embedding(edge_feature_size, gnn_hidden_dim),
            'convs': nn.ModuleList([
                GATConv(gnn_hidden_dim, gnn_hidden_dim // gnn_num_heads, heads=gnn_num_heads, dropout=gnn_dropout)
                for _ in range(gnn_num_layers)
            ]),
            'bns': nn.ModuleList([nn.BatchNorm1d(gnn_hidden_dim) for _ in range(gnn_num_layers)]),
            'dropout': nn.Dropout(gnn_dropout)
        })

        self.regressor = nn.ModuleDict({
            'input_layer': nn.Sequential(
                nn.Linear(gnn_hidden_dim * 2 + 1, regressor_hidden_dim),
                nn.BatchNorm1d(regressor_hidden_dim),
                nn.ReLU(),
                nn.Dropout(regressor_dropout)
            ),
            'hidden_layers': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(regressor_hidden_dim, regressor_hidden_dim),
                    nn.BatchNorm1d(regressor_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(regressor_dropout)
                )
                for _ in range(regressor_num_hidden_layers)
            ]),
            'output_layer': nn.Linear(regressor_hidden_dim, 1)
        })

    def forward(self, x, edge_index, edge_attr, batch, N, descriptors=None):
        """Forward pass (descriptors ignored for compatibility)."""
        embed_indices = x[:, 0].long().squeeze()
        other_node_features = x[:, 1:].float().squeeze()
        embedded_node = self.gnn['node_embedding'](embed_indices)
        processed_other_features = self.gnn['lin_node_features'](other_node_features)
        h = torch.cat([embedded_node, processed_other_features], dim=1)

        edge_attr_indices = edge_attr[:, 0].long().squeeze()
        edge_attr_embedded = self.gnn['edge_embedding'](edge_attr_indices)

        for i in range(len(self.gnn['convs'])):
            h = self.gnn['convs'][i](h, edge_index, edge_attr=edge_attr_embedded)
            h = self.gnn['bns'][i](h)
            h = F.relu(h)
            h = self.gnn['dropout'](h)

        graph_embedding = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        graph_embedding = torch.cat([graph_embedding, N.unsqueeze(1)], dim=1)

        x_reg = self.regressor['input_layer'](graph_embedding)
        for layer in self.regressor['hidden_layers']:
            x_reg = layer(x_reg)
        prediction = self.regressor['output_layer'](x_reg)
        return prediction

