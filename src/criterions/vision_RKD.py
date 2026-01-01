import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from .utils import get_grid_size, get_hidden_text_vision

class VisionRKDLoss(nn.Module):
    def __init__(self, args):
        super(VisionRKDLoss, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_loss_weight = self.args.kd_weight
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        student_processor = distiller.get_student_processor()
        teacher_processor = distiller.get_teacher_processor()
        student_tokenizer = student_processor.tokenizer
        teacher_tokenizer = teacher_processor.tokenizer
        

        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
        
        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output
        
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)
        
        loss_distill = 0.0
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0
        loss_vsd = 0.0
        loss_vlad = 0.0

        student_special_ids = torch.tensor(student_tokenizer.all_special_ids, device=student_qry_input['input_ids'].device)
        teacher_special_ids = torch.tensor(teacher_tokenizer.all_special_ids, device=teacher_qry_input['input_ids'].device)

        num_student_text_qry_tokens = (~torch.isin(student_qry_input['input_ids'], 
                                                   student_special_ids)).sum(dim=1)
        num_student_text_pos_tokens = (~torch.isin(student_pos_input['input_ids'], 
                                                   student_special_ids)).sum(dim=1)

        num_teacher_text_qry_tokens = (~torch.isin(teacher_qry_input['input_ids'], 
                                                   teacher_special_ids)).sum(dim=1)
        num_teacher_text_pos_tokens = (~torch.isin(teacher_pos_input['input_ids'], 
                                                   teacher_special_ids)).sum(dim=1)
        
        stu_qry_vision_grid_sizes = get_grid_size(student_model, student_qry_input)
        tea_qry_vision_grid_sizes = get_grid_size(teacher_model, teacher_qry_input)

        stu_pos_vision_grid_sizes = get_grid_size(student_model, student_pos_input)
        tea_pos_vision_grid_sizes = get_grid_size(teacher_model, teacher_pos_input)
        
        for i in range(batch_size):
            # print(f"Sample {i}: num_text_qry_tokens {num_text_qry_tokens[i]}, num_text_pos_tokens {num_text_pos_tokens[i]}")
            # print(f"Sample {i} input_ids ids of teacher {teacher_qry_input['input_ids'][i]}, pos {teacher_pos_input['input_ids'][i]}")
            # print(f"Sample {i} input_ids ids of student {student_qry_input['input_ids'][i]}, pos {student_pos_input['input_ids'][i]}")
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    if student_qry_image_features[cur_idx_qry_img] is not None and teacher_qry_image_features[cur_idx_qry_img] is not None:
                        num_tokens_vision_qry_stu = student_qry_image_features[cur_idx_qry_img].size(0)
                        num_tokens_vision_qry_tea = teacher_qry_image_features[cur_idx_qry_img].size(0)

                        student_qry_text_hidden_state, last_stu_qry_vision_hidden_state  = get_hidden_text_vision(
                            student_qry_hidden_states[-1][i], 
                            num_student_text_qry_tokens[i].item(), 
                            num_tokens_vision_qry_stu, 
                            student_model.model_backbone
                        )

                        teacher_qry_text_hidden_state, last_tea_qry_vision_hidden_state  = get_hidden_text_vision(
                            teacher_qry_hidden_states[-1][i], 
                            num_teacher_text_qry_tokens[i].item(), 
                            num_tokens_vision_qry_tea, 
                            teacher_model.model_backbone
                        )

                        last_stu_qry_vision_hidden_state_norm = F.normalize(last_stu_qry_vision_hidden_state, p=2, dim=-1) # (N_s, D)

                        projected_last_tea_qry_vision_hidden_state = self.distiller.projectors["t2s"](last_tea_qry_vision_hidden_state) # (N_t, D)
                        last_tea_qry_vision_hidden_state_norm = F.normalize(projected_last_tea_qry_vision_hidden_state, p=2, dim=-1) # (N_t, D)

                        c2 = 1.0 - last_stu_qry_vision_hidden_state_norm @ last_tea_qry_vision_hidden_state_norm.T # (N_s, N_t)

        loss = contrastive_loss

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            # 'kd_loss': loss_distill
        }