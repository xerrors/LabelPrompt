# def get_pre_embeddings(self, input_embeddings, attention_mask, so, return_ent_emb=False):
#     """ 废弃 """
#     if self.args.use_pre_prompt:

#         sub_inputs = input_embeddings.clone()
#         obj_inputs = input_embeddings.clone()

#         for i in range(len(so)):
#             sub_st, sub_ed = so[i][1]+2, so[i][4]-1
#             sub_token = torch.arange(sub_st, sub_ed).cuda()
#             sub_inputs[i, sub_st:sub_ed] = self.pre_encoder(sub_token).unsqueeze(0)

#             obj_st, obj_ed = 1, so[i][2]-1
#             obj_token = torch.arange(obj_st, obj_ed).cuda()
#             obj_inputs[i, obj_st:obj_ed] = self.pre_encoder(obj_token).unsqueeze(0)

#         sub_emb = self.pre_model(inputs_embeds=sub_inputs, attention_mask=attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1][:, 0]
#         obj_emb = self.pre_model(inputs_embeds=obj_inputs, attention_mask=attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1][:, 0]

#     else:
#         sub_mask = attention_mask.clone()
#         obj_mask = attention_mask.clone()

#         for i in range(len(so)):
#             sub_mask[i, so[i][1]+2:so[i][4]-1] = 0
#             obj_mask[i, 1:so[i][2]-1] = 0

#         sub_emb = self.pre_model(inputs_embeds=input_embeddings, attention_mask=sub_mask, return_dict=True, output_hidden_states=True).hidden_states[-1][:, 0]
#         obj_emb = self.pre_model(inputs_embeds=input_embeddings, attention_mask=obj_mask, return_dict=True, output_hidden_states=True).hidden_states[-1][:, 0]

#     for i in range(len(so)):
#         input_embeddings[i, so[i][0]-1] = sub_emb[i]
#         input_embeddings[i, so[i][2]-1] = obj_emb[i]

#     if return_ent_emb:
#         return input_embeddings, sub_emb, obj_emb
#     else:
#         return input_embeddings