# TAG: Make Cloze
# copy from github.com/google-reserch/bert : https://github.com/google-research/bert/blob/master/create_pretraining_data.py
import collections
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def make_cloze(tokenizer, tokens, pos, rng, vocab_words,
                do_whole_word_mask=False, max_predictions_per_seq=20,
                masked_lm_prob=0.15, predefined_cloze=3):
    """
        pos: [sep, real_token_len]
    """
    sep, real_token_len = pos

    t_debug = tokenizer.all_special_tokens

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if i <= sep or i >= real_token_len or token in tokenizer.all_special_ids:
            continue
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(real_token_len * masked_lm_prob))))

    # FIXME: set num_to_predict woth const 3 for debug
    num_to_predict = predefined_cloze
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token_id = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token_id = tokenizer.convert_tokens_to_ids(["[cloze]"])[0]
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token_id = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token_id = tokenizer.convert_tokens_to_ids([vocab_words[rng.randint(0, len(vocab_words) - 1)]])[0]
            output_tokens[index] = masked_token_id
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    assert len(masked_lms) == num_to_predict  # for debug
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)