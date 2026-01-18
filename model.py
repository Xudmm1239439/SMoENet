import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, ceil
#from timm.models.vision_transformer import Block


class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target, mask):
        mask_ = mask.view(-1, 1)
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)   
        return loss


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())  
        return loss

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class MultiHeadedAttention1(nn.Module):
    def __init__(self, head_count: int, model_dim: int, dropout: float = 0.1,topk=0.7):
        super(MultiHeadedAttention1, self).__init__()
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.topk = topk
        self.head_count = head_count

        # 线性变换层
        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.norm = nn.LayerNorm(model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

        # 初始化权重
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, key, value, query, mask = None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        topk = self.topk

        def shape(x):
            """投影"""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """计算上下文"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 应用线性变换
        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        
        _, _,C, _ = query.shape
        # 缩放查询
        query = query / math.sqrt(dim_per_head)
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(2, 3))
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, float('-inf'))

        # 应用Top-K稀疏化
        # 创建四个Top-K掩码
        mask1 = torch.zeros(batch_size, head_count, C, C, device=key.device, dtype=torch.bool)
        
        # 选择Top-K连
        values, indices = torch.topk(scores, k=int(C*(topk)), dim=-1, largest=True)    
        mask1.scatter_(dim=-1, index=indices, value=True)
        
        # 应用Top-K掩码
        scores1 = torch.where(mask1, scores, torch.full_like(scores, float('-inf')))
        
        # 计算上下文
        attn1 = self.softmax(scores1)
        
        # 生成输出
        drop_attn = self.dropout(attn1)
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output
    

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    


class Block(nn.Module):
    def __init__(
            self,dim: int,topk,num_heads: int,init_values: float = None,dropout=0.5
    ) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.attn = MultiHeadedAttention1(num_heads, dim, dropout,topk)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
       
    def MLP(self, features):
        feat = self.fc2(self.dropout1(nn.GELU()(self.fc1(features))))
        return self.dropout2(feat)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = x + self.ls1(self.attn(x,x,x))
        x = x + self.ls2(self.MLP(self.norm2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)
        
        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == 'MELD':
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep

class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep
    

class TokenAttention(torch.nn.Module):
    """
    Compute attention layer for 3-dimensional feature inputs
    """
    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_shape, input_shape),
            nn.SiLU(),
            nn.Linear(input_shape, 1),
        )
        self.input_projection = nn.Linear(3 * input_shape, input_shape)

    def forward(self, inputs):
        """
        inputs: [batch, seq_len, 3, model_dim]
        """
        scores = self.attention_layer(inputs).view(inputs.size(0),inputs.size(1),-1)
        scores = scores.unsqueeze(2)
        # 计算注意力分数，并调整形状
        outputs = torch.matmul(scores, inputs).squeeze(2)
        return outputs, scores
    
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=False)

    def forward(self, t, a,v):
        eps = 1e-10
        t, a = t + eps, a + eps
        v = v+eps
        t = t / t.sum(dim=-1, keepdim=True)
        a = a / a.sum(dim=-1, keepdim=True)
        v = v / v.sum(dim=-1, keepdim=True)
        
        # Applying a small epsilon to avoid log(0)
        m_ta = 0.5 * (t + a)
        log_m_ta = m_ta.log()
        kl_t_m_ta = self.kl(log_m_ta, t.log())
        kl_a_m_ta = self.kl(log_m_ta, a.log())
        jsd_ta = 0.5 * (kl_t_m_ta + kl_a_m_ta).sum(dim=-1)

        m_tv = 0.5 * (t + v)
        log_m_tv = m_tv.log()
        kl_t_m_tv = self.kl(log_m_tv, t.log())
        kl_v_m_tv = self.kl(log_m_tv, v.log())
        jsd_tv = 0.5 * (kl_t_m_tv + kl_v_m_tv).sum(dim=-1)

        m_av = 0.5 * (a + v)
        log_m_av = m_av.log()
        kl_a_m_av = self.kl(log_m_av, a.log())
        kl_v_m_av = self.kl(log_m_av, v.log())
        jsd_av = 0.5 * (kl_a_m_av + kl_v_m_av).sum(dim=-1)

        return jsd_ta,jsd_tv,jsd_av

class InteractionModule(nn.Module):
    def __init__(self, hidden_dim, gamma):
        agr_threshold = 0.3
        
        balance_loss_coef = 0.0
        router_z_loss_coef = gamma#0.00434#0.194#i0.01#m0.017#0.00434
        interaction_loss_coef = 0.7
        super(InteractionModule, self).__init__()
        self.jsd_module = JSD()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)
        # 256
        self.hidden_dim = 1024
        self.modality_attn = TokenAttention(self.hidden_dim)
        self.soft_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),#4是类别
        )

        self.agr_threshold = torch.tensor(agr_threshold, requires_grad=False)
        self.interaction_loss = nn.CrossEntropyLoss()
        self.interaction_loss_coef = interaction_loss_coef
        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef


    def compute_router_z_loss(self, gate1_logits, gate2_logits, gate3_logits):
        logits = torch.cat([gate1_logits, gate2_logits, gate3_logits], dim=1)
        max_logits = torch.logsumexp(logits, dim=1)
        router_z_loss = torch.mean(max_logits**2)
        return self.router_z_loss_coef * router_z_loss

    def compute_balance_loss(self, assignments):
        num_experts = 2
        batch_size = assignments.size(0)
        expert_counts = torch.zeros(num_experts)
        for i in range(num_experts):
            expert_counts[i] = (assignments == i).sum()
        distribution = expert_counts / batch_size
        target_distribution = torch.full_like(distribution, fill_value=1 / num_experts)
        balancing_loss = F.mse_loss(distribution, target_distribution)
        return balancing_loss
    

    def forward(self, p_t, p_a, p_v, e_t, e_a, e_v):
       
        jsd_ta,jsd_tv,jsd_av = self.jsd_module(p_t, p_a,p_v)
        
        agr_gate_scores_ta = (jsd_ta < self.agr_threshold).type(torch.int64)
        agr_gate_scores_tv = (jsd_tv < self.agr_threshold).type(torch.int64)
        stacked_features = torch.stack(
            (e_t, e_a, e_v),
            dim=2,
        )
        
        gate_inputs, _ = self.modality_attn(stacked_features)     
        gate_logits = self.soft_gate(gate_inputs)      
        targets = agr_gate_scores_ta+agr_gate_scores_tv
        agr_logits = gate_logits[:, :,:1]
        nagr_logits1 = gate_logits[:, :,1:2]
        nagr_logits2 = gate_logits[:, :,2:]
        
        
        agr_gate = torch.argmax(F.softmax(agr_logits, dim=1), dim=1)
        nagr_gate1 = torch.argmax(F.softmax(nagr_logits1, dim=1), dim=1)
        nagr_gate2 = torch.argmax(F.softmax(nagr_logits2, dim=1), dim=1)
        dispatch_index = agr_gate + nagr_gate1+ nagr_gate2# + sem_gate
        router_z_loss = self.router_z_loss_coef * self.compute_router_z_loss(
            agr_logits, nagr_logits1,nagr_logits2#, sem_logits
        )
        balance_loss = self.balance_loss_coef * self.compute_balance_loss(
            dispatch_index
        )
        expert_mask = F.softmax(gate_logits, dim=1)
        gate_loss =  router_z_loss
        #gate_loss = router_z_loss 
        return expert_mask, gate_loss





class Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout,gamma,topk):
        super(Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.gamma=gamma
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        
        # Temporal convolutional layers
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
        
        # Intra- and Inter-modal Transformers
        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        
        # Unimodal-level Gated Fusion
        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        # Multimodal-level Gated Fusion
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Emotion Classifier
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.all_output_layer = nn.Linear(hidden_dim, n_classes)



        self.depth = 1  # 2
        self.num_expert = 2  # 2
        self.mix_trim = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.t_trim = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.a_trim = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.v_trim = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.interaction_module = InteractionModule(
            self.hidden_dim,self.gamma
        )

        self.final_attention = nn.ModuleList(
            [TokenAttention(self.hidden_dim) for i in range(3)]
        )
        self.fusion_SE_network_main_task = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.num_expert),
                )
                for i in range(3)
            ]
        )

        def _expert():
            fusing_expert_ls = []
            for i in range(self.num_expert):
                fusing_expert = []
                for j in range(self.depth):
                    fusing_expert.append(Block(dim=self.hidden_dim,topk=self.topk,num_heads=8))
                fusing_expert = nn.ModuleList(fusing_expert)
                fusing_expert_ls.append(fusing_expert)
            return nn.ModuleList(fusing_expert_ls)
        self.final_fusing_experts = nn.ModuleList([_expert() for i in range(3)])

    

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        spk_embeddings = self.speaker_embeddings(spk_idx)

        # Temporal convolutional layers
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        # Intra- and Inter-modal Transformers
        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings)
        v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings)

        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings)
        v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings)

        v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)
        t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings)
        a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings)

        # Unimodal-level Gated Fusion
        t_t_transformer_out = self.t_t_gate(t_t_transformer_out)
        a_t_transformer_out = self.a_t_gate(a_t_transformer_out)
        v_t_transformer_out = self.v_t_gate(v_t_transformer_out)

        a_a_transformer_out = self.a_a_gate(a_a_transformer_out)
        t_a_transformer_out = self.t_a_gate(t_a_transformer_out)
        v_a_transformer_out = self.v_a_gate(v_a_transformer_out)

        v_v_transformer_out = self.v_v_gate(v_v_transformer_out)
        t_v_transformer_out = self.t_v_gate(t_v_transformer_out)
        a_v_transformer_out = self.a_v_gate(a_v_transformer_out)

        t_transformer_out = self.features_reduce_t(torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1))
        v_transformer_out = self.features_reduce_v(torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1))

        # Emotion te
        t_te_out = self.t_trim(t_transformer_out)
        a_te_out = self.a_trim(a_transformer_out)
        v_te_out = self.v_trim(v_transformer_out)

        # Emotion Classifier
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)

        expert_mask, interaction_loss = self.interaction_module(
            t_final_out,
            a_final_out,
            v_final_out,
            t_te_out,
            a_te_out,
            v_te_out,
        )
        # # Multimodal-level Gated Fusion
        # all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)
        all_transformer_out = torch.zeros_like(t_transformer_out)
        for k in range(3):#路由数量
            # 调整输入特征的维度以适应序列输入：[batch, seq_len, model_dim]
            concat_feature_main_biased = torch.stack((t_transformer_out, a_transformer_out, v_transformer_out), dim=2)  # [batch, seq_len, 3 , model_dim]
            #print("concat_feature_main_biased.shape",concat_feature_main_biased.shape)
            
            # # 创建mask（如果需要根据具体任务来设计mask）
            # mask = expert_mask[:, k].unsqueeze(1)  # [batch, 1]
            
            # 通过注意力机制融合特征
            fusion_tempfeat_main_task, _ = self.final_attention[k](concat_feature_main_biased)  # [batch, seq_len, model_dim]
            #print("fusion_tempfeat_main_task.shape",fusion_tempfeat_main_task.shape)
            # SE网络融合任务特征
            gate_main_task = self.fusion_SE_network_main_task[k](fusion_tempfeat_main_task)  # [batch, seq_len, num_expert]
            #print("gate_main_task.shape",gate_main_task.shape)
            #mask = torch.ones_like(gate_main_task[:, :, 0]).unsqueeze(2)  # [batch, seq_len] 全1张量，每个专家平等
            mask = expert_mask[:, :,k].unsqueeze(2)
            
            
            concat_feature_main_biased1 = t_transformer_out + a_transformer_out + v_transformer_out
            # 初始化输出任务特征
            final_feature_main_task = torch.zeros_like(concat_feature_main_biased1)  # [batch, seq_len, model_dim]
            
            #print("concat_feature_main_biased1.shape",concat_feature_main_biased1.shape)
            # 融合专家部分
            for i in range(self.num_expert):
                fusing_expert = self.final_fusing_experts[k][i]
                tmp_fusion_feature = concat_feature_main_biased1  # [batch, seq_len, 3 * model_dim]
                
                # 对每个深度层应用专家网络
                for j in range(self.depth):
                    tmp_fusion_feature = fusing_expert[j](tmp_fusion_feature )  # [batch, seq_len, model_dim]
                    #print("tmp_fusion_feature.shape",tmp_fusion_feature.shape)
                # # 取每个时间步的特征（选择[0]表示选择每个时间步的输出）
                # tmp_fusion_feature = tmp_fusion_feature[:, 0]  # [batch, model_dim]
                # print("tmp_fusion_feature.shape",tmp_fusion_feature.shape)
                # print("gate_main_task[:, :,i].unsqueeze(2).shape",gate_main_task[:, :,i].unsqueeze(2).expand((-1, -1, 1024)) .shape)
                gate_main = gate_main_task[:,:, i].unsqueeze(2)
                # print("tmp_fusion_feature.shape",tmp_fusion_feature.shape)
                # print("gate_main.shape",gate_main.shape)
                # 按照门控机制加权融合
                final_feature_main_task += tmp_fusion_feature * gate_main   # [batch, seq_len, model_dim]
            
            # 混合（可以是一些处理特征的操作，比如降维或池化）
            final_feature_main_task_lite = self.mix_trim(final_feature_main_task)  # [batch, seq_len, model_dim]
            
            # 将mask应用到最终的特征输出（可以通过逐元素乘法）
            #print("final_feature_main_task_lite.shape",final_feature_main_task_lite.shape)
            #print("mask.shape",mask.shape)
            all_transformer_out += final_feature_main_task_lite * mask  # [batch, seq_len, model_dim]


        #print("all_transformer_out.shape",all_transformer_out.shape)
        all_final_out = self.all_output_layer(all_transformer_out)

        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        kl_t_log_prob = F.log_softmax(t_final_out /self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out /self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out /self.temp, 2)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)

        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
               kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob,interaction_loss,all_transformer_out
