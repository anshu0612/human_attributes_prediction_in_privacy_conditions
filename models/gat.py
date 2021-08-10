import torch
import torch.nn as nn
import torch.nn.functional as F


class _GraphAttentionLayer(nn.Module):
    '''
    Part of code borrows from  https://github.com/Diego999/pyGAT/blob/master/layers.py
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    '''

    def __init__(self, in_feat, out_feat, dropout, alpha, concat=True):
        super(_GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_feat, out_feat)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_feat, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[1]
        batch = h.size()[0]
        a_input = torch.cat([h.repeat(1, 1, N).view(
            batch, N * N, -1), h.repeat(1, N, 1)], dim=2).view(batch, N, -1, 2 * self.out_feat)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_feat) + ' -> ' + str(self.out_feat) + ')'


class _GAT(nn.Module):
    '''
    Dense version of GAT.
    '''

    def __init__(self, num_feat, num_hidd, num_class, dropout, alpha, num_heads):
        super(_GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [_GraphAttentionLayer(
            num_feat, num_hidd, dropout=dropout, alpha=alpha, concat=True) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = _GraphAttentionLayer(
            num_hidd * num_heads, num_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=2)


class VisualRelationshipStream(nn.Module):
    def __init__(self, device):
        super(VisualRelationshipStream, self).__init__()

        _context_reshape_size = 180

        self.context_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(_context_reshape_size),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.target_feat = nn.Sequential(
            nn.AdaptiveMaxPool2d((60, 15)),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self._patch_size = 30
        _n_patches = (_context_reshape_size // self._patch_size)
        _nodes = (_n_patches * _n_patches) + 1  # extra node for the target
        _gat_in_feat = self._patch_size * self._patch_size

        self.gat = _GAT(num_feat=_gat_in_feat,
                        num_hidd=128,
                        num_class=256,
                        dropout=0.5,
                        num_heads=3,
                        alpha=0.2)

        self.adj = torch.ones([_nodes, _nodes], dtype=torch.int32).to(device)

    def _generate_image_pataches(self, size, images):
        _channel_dim = images.size()[1]
        patches = images.unfold(1, _channel_dim, _channel_dim).unfold(
            2, size, size).unfold(3, size, size)
        s_ = patches.size()

        # batch_size, noedes - 36,   00
        patches = patches.reshape(s_[0], s_[1]*s_[2]*s_[3], s_[4]*s_[5]*s_[6])
        return patches

    def forward(self, context, target):
        _batch = context.size()[0]

        context = self.context_pool(context)
        patched_context = self._generate_image_pataches(
            self._patch_size, context)

        target = self.target_feat(target)
        target = target.view(_batch, -1)
        target = target.unsqueeze(1)

        patched_context = torch.cat((patched_context, target), 1)
        patched_context = self.gat(patched_context, self.adj)
        patched_context = torch.mean(patched_context, 1)

        return patched_context
