import torch
import random
import torch.nn.functional as F

def type_loss(lit_model, hidden_state, logits, labels, so, input_ids, attention_mask):
    bsz = hidden_state.shape[0]
    weights = lit_model.model.get_output_embeddings().weight

    mask_idx = (input_ids == lit_model.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # current just use embeddings with any projection.
    prefix_type = torch.stack([torch.mean(weights[lit_model.rel_subtype[labels[i].item()][0]], dim=0) for i in range(bsz)])
    suffix_type = torch.stack([torch.mean(weights[lit_model.rel_subtype[labels[i].item()][1]], dim=0) for i in range(bsz)])

    # default use mask tags
    if lit_model.args.init_rel_subtype == "use_entity_tags":
        mark_prefix = torch.stack([hidden_state[i, so[i][0]] for i in range(bsz)]).cuda()     # [sub] tag before subject in prompt template
        mark_suffix = torch.stack([hidden_state[i, so[i][2]] for i in range(bsz)]).cuda()     # [obj] tag before subject in prompt template

    elif lit_model.args.init_rel_subtype == "use_logits_top":
        prefix_top10 = torch.topk(lit_model.model.lm_head(prefix_type), k=10, dim=-1)
        suffix_top10 = torch.topk(lit_model.model.lm_head(suffix_type), k=10, dim=-1)

        mark_prefix_logits = logits[torch.arange(bsz), mask_idx-1]
        mark_suffix_logits = logits[torch.arange(bsz), mask_idx+2]
        mark_prefix = mark_prefix_logits[torch.arange(bsz), prefix_top10.indices.T].T
        mark_suffix = mark_suffix_logits[torch.arange(bsz), suffix_top10.indices.T].T

        prefix_type = prefix_top10.values
        suffix_type = suffix_top10.values

    else:
        mark_prefix = hidden_state[torch.arange(bsz), mask_idx-1]   # [m] tag
        mark_suffix = hidden_state[torch.arange(bsz), mask_idx+2]   # [/m] tag

    if lit_model.args.use_type_projection and lit_model.args.init_rel_subtype == "use_logits_top":
        raise NameError("can not use `use_type_projection` and `init_rel_subtype` at the same time.")

    if lit_model.args.use_type_projection and lit_model.args.init_rel_subtype != "use_logits_top":
        prefix_type = lit_model.type_projection(prefix_type)
        suffix_type = lit_model.type_projection(suffix_type)
        mark_prefix = lit_model.type_projection(mark_prefix)
        mark_suffix = lit_model.type_projection(mark_suffix)

    # similarity func
    similarity_funcs = {
        "cosine": lambda a, b: 1 - torch.cosine_similarity(a, b, dim=-1),
        "l2": lambda a, b: torch.norm(a-b, p=2, dim=-1)
    }

    similarity_func = similarity_funcs[lit_model.args.similarity_func]
    loss = similarity_func(mark_prefix, prefix_type).sum() + similarity_func(mark_suffix, suffix_type).sum()

    return loss / bsz


# Loss - Cloze Loss
def cloze_loss(lit_model, logits, masked_lm_positions, masked_lm_labels, input_ids):
    """
    Inputs:
        - logits                 => [batch_size, seq_len, vocab_size]
        - masked_lm_positions    => [batch_size, cloze_size]
        - masked_lm_labels       => [batch_size, cloze_size]
        - input_ids              => [batch_size, seq_len]

    Targets:
        [batch_size, seq_len, vocab_size] => [batch_size, cloze_size, vocab_size] => [batch_size, cloze_size]
    """
    bs, seq_len, vocab_size = logits.shape
    ont_hot_label = F.one_hot(masked_lm_labels, num_classes=vocab_size)  # => [batch_size, cloze_size, vocab_size]

    log_probs = F.log_softmax(logits, dim=-1)  # => [batch_size, seq_len, vocab_size]

    loss = 0
    for i in range(bs):
        temp = log_probs[i, masked_lm_positions[i], :]  * ont_hot_label[i]  # => [cloze_size, vocab_size]
        temp = torch.sum(temp, dim=-1).squeeze(-1)  # => [cloze_size]
        temp = torch.sum(temp, dim=-1).squeeze(-1)  # => scalar
        loss += temp / bs

    return -1 * loss

# Loss - KE Loss
def ke_loss(lit_model, hidden_state, labels, so, input_ids, attention_mask):
    """ Implicit Structured Constraints """
    bsz = hidden_state.shape[0]

    assert len(lit_model.args.ke_type.split(":")) == 2
    ke_type, calc_type = lit_model.args.ke_type.split(":")

    pos_sub_embdeeings, pos_obj_embdeeings, neg_sub_embeddings, neg_obj_embeddings = get_entity_embeddings(hidden_state, so, bsz, pos=True, neg=True)
    # trick , the relation ids is concated,

    _, mask_idx = (input_ids == lit_model.tokenizer.mask_token_id).nonzero(as_tuple=True)
    mask_output = hidden_state[torch.arange(bsz), mask_idx]
    mask_relation_embedding = mask_output
    real_relation_embedding = lit_model.model.get_output_embeddings().weight[labels+lit_model.label_st_id]

    log_sigmoid = torch.nn.LogSigmoid()

    if lit_model.args.use_emb_projection:
        pos_sub_embdeeings = lit_model.projection_sub(pos_sub_embdeeings)
        pos_obj_embdeeings = lit_model.projection_obj(pos_obj_embdeeings)
        neg_sub_embeddings = lit_model.projection_sub(neg_sub_embeddings)
        neg_obj_embeddings = lit_model.projection_obj(neg_obj_embeddings)
        mask_relation_embedding = lit_model.projection_rel(mask_relation_embedding)
        real_relation_embedding = lit_model.projection_rel(real_relation_embedding)

    if calc_type == "default":
        d_1 = torch.norm(pos_sub_embdeeings + mask_relation_embedding - pos_obj_embdeeings, p=2) / bsz
        d_2 = torch.norm(neg_sub_embeddings + real_relation_embedding - neg_obj_embeddings, p=2) / bsz

    elif calc_type == "mask":
        d_1 = torch.norm(pos_sub_embdeeings + mask_relation_embedding - pos_obj_embdeeings, p=2) / bsz
        d_2 = torch.norm(neg_sub_embeddings + mask_relation_embedding - neg_obj_embeddings, p=2) / bsz

    elif calc_type == "mask_linear":
        d_1 = torch.norm(lit_model.linear_transe(torch.cat([pos_sub_embdeeings, mask_relation_embedding, pos_obj_embdeeings], dim=-1)), p=2) / bsz
        d_2 = torch.norm(lit_model.linear_transe(torch.cat([neg_sub_embeddings, mask_relation_embedding, neg_obj_embeddings], dim=-1)), p=2) / bsz

    elif calc_type == "default_linear":
        d_1 = torch.norm(lit_model.linear_transe(torch.cat([pos_sub_embdeeings, mask_relation_embedding, pos_obj_embdeeings], dim=-1)), p=2) / bsz
        d_2 = torch.norm(lit_model.linear_transe(torch.cat([neg_sub_embeddings, real_relation_embedding, neg_obj_embeddings], dim=-1)), p=2) / bsz
    else:
        raise NameError("Unable to recogniz calc_type: '{}' or 'mask_reverse' can be used only if the corpus_type equals to 'semeval'".format(calc_type))

    loss = -1. * log_sigmoid(lit_model.args.t_gamma - d_1) - log_sigmoid(d_2 - lit_model.args.t_gamma)

    return loss

def get_entity_embeddings(hidden_state, so, bsz, pos=True, neg=False):
    pos_sub_embdeeings = []
    pos_obj_embdeeings = []

    for i in range(bsz):
        sub_e = torch.mean(hidden_state[i, so[i][0]:so[i][1]], dim=0)  # include "space"
        obj_e = torch.mean(hidden_state[i, so[i][2]:so[i][3]], dim=0)  # include "space"
        pos_sub_embdeeings.append(sub_e)
        pos_obj_embdeeings.append(obj_e)

    pos_sub_embdeeings = torch.stack(pos_sub_embdeeings)
    pos_obj_embdeeings = torch.stack(pos_obj_embdeeings)

    neg_sub_embeddings = []
    neg_obj_embeddings = []
    for i in range(bsz):
        st_sub = random.randint(1, hidden_state[i].shape[0] - 6)
        st_obj = random.randint(1, hidden_state[i].shape[0] - 6)
        neg_sub_e = torch.mean(hidden_state[i, st_sub:st_sub + random.randint(1, 5)], dim=0)
        neg_obj_e = torch.mean(hidden_state[i, st_obj:st_obj + random.randint(1, 5)], dim=0)

        neg_sub_embeddings.append(neg_sub_e)
        neg_obj_embeddings.append(neg_obj_e)

    neg_sub_embeddings = torch.stack(neg_sub_embeddings)
    neg_obj_embeddings = torch.stack(neg_obj_embeddings)

    if pos and neg:
        return pos_sub_embdeeings, pos_obj_embdeeings, neg_sub_embeddings, neg_obj_embeddings
    elif pos:
        return pos_sub_embdeeings, pos_obj_embdeeings
    elif neg:
        return neg_sub_embeddings, neg_obj_embeddings

def pre_ke(lit_model, input_ids, hidden_state, pos_sub_embdeeings, pos_obj_embdeeings, labels, so):

    bsz = hidden_state.shape[0]

    _, mask_idx = (input_ids == lit_model.tokenizer.mask_token_id).nonzero(as_tuple=True)
    mask_output = hidden_state[torch.arange(bsz), mask_idx]
    mask_relation_embedding = mask_output
    real_relation_embedding = lit_model.model.get_output_embeddings().weight[labels+lit_model.label_st_id]

    neg_sub_embeddings, neg_obj_embeddings = get_entity_embeddings(hidden_state, so, bsz, False, True)

    d_1 = torch.norm(lit_model.linear_transe(torch.cat([pos_sub_embdeeings, mask_relation_embedding, pos_obj_embdeeings], dim=-1)), p=2) / bsz
    d_2 = torch.norm(lit_model.linear_transe(torch.cat([neg_sub_embeddings, real_relation_embedding, neg_obj_embeddings], dim=-1)), p=2) / bsz

    log_sigmoid = torch.nn.LogSigmoid()
    loss = -1. * log_sigmoid(lit_model.args.t_gamma - d_1) - log_sigmoid(d_2 - lit_model.args.t_gamma)

    return loss