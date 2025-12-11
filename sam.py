# /workspace/FAS_ICCV/sam.py
import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # base_optimizer는 클래스 자체 (예: torch.optim.AdamW) 또는 이미 초기화된 옵티마이저 인스턴스를 받을 수 있습니다.
        # 여기서는 base_optimizer 클래스를 받아 내부에서 초기화하는 형태로 되어 있습니다.
        # **kwargs에는 base_optimizer에 필요한 인자들(lr, weight_decay 등)이 전달됩니다.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups # SAM이 base_optimizer의 param_groups를 공유하도록 설정
        self.defaults.update(self.base_optimizer.defaults) # SAM의 defaults에도 base_optimizer의 설정을 반영

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone() # 원래 파라미터 저장
                e_w = (torch.pow(p.data, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p.device) # adaptive SAM 여부
                p.add_(e_w)  # 파라미터를 그래디언트 방향으로 이동 (손실 증가 방향)

        if zero_grad: self.zero_grad() # 다음 backward를 위해 그래디언트 초기화

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue # 두 번째 backward에서 grad가 없는 경우 스킵
                p.data = self.state[p]["old_p"]  # 파라미터를 원래 위치로 복원

        self.base_optimizer.step()  # 복원된 파라미터 위치에서 base_optimizer로 실제 업데이트 수행

        if zero_grad: self.zero_grad()

    # step 함수는 스케줄러 호환성을 위해 존재할 수 있으나, SAM의 핵심 로직은 first_step과 second_step에 있습니다.
    # 직접 호출될 일은 거의 없습니다.
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("SAM.step(closure) is not implemented.")
        # 일반적으로 SAM은 first_step과 second_step을 통해 업데이트가 이루어지므로,
        # 이 step 함수는 직접적인 업데이트 로직을 포함하지 않을 수 있습니다.
        # 만약 스케줄러가 optimizer.step()을 호출한다면, base_optimizer의 상태를 업데이트하기 위해
        # self.base_optimizer.step()을 호출하는 것을 고려할 수 있으나, 이는 SAM의 의도와 다를 수 있습니다.
        # 현재 구현에서는 second_step에서 base_optimizer.step()이 호출되므로,
        # 이 step()은 사실상 아무것도 하지 않거나, 오류를 발생시키는 것이 더 명확할 수 있습니다.
        # print("Warning: SAM.step() called. This typically should not happen directly. Ensure scheduler uses base_optimizer if needed.")

    def _grad_norm(self):
        # Ensure all computations are on the same device
        # Handle cases where params might be on different devices (model parallelism)
        # For simplicity, assume all params in a group are on the same device as the first param.
        # More robust: iterate and move norms to a common device before stacking.
        shared_device = self.param_groups[0]["params"][0].device
        
        all_norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_norm = p.grad.norm(p=2)
                if group["adaptive"]:
                    param_norm = (torch.abs(p.data) * p.grad).norm(p=2) # Corrected: use p.data for adaptive weight
                all_norms.append(param_norm.to(shared_device))
        
        if not all_norms: # Handle case with no gradients
            return torch.tensor(0.0, device=shared_device)

        norm = torch.norm(torch.stack(all_norms), p=2)
        return norm

    def load_state_dict(self, state_dict):
        # SAM의 상태와 base_optimizer의 상태를 모두 로드/저장해야 합니다.
        # state_dict는 SAM의 상태와 base_optimizer의 상태를 포함할 수 있습니다.
        # PyTorch의 Optimizer.load_state_dict는 param_groups를 직접 다루지 않으므로,
        # base_optimizer의 param_groups가 SAM의 것과 동기화되도록 해야 합니다.
        super().load_state_dict(state_dict) # SAM 자체의 상태 (rho 등) 로드
        # base_optimizer의 상태는 state_dict['base_optimizer_state_dict'] 같은 키로 저장되어 있을 수 있습니다.
        # 또는, base_optimizer.load_state_dict를 직접 호출해야 할 수도 있습니다.
        # 현재 구현은 base_optimizer가 SAM의 param_groups를 사용하도록 __init__에서 설정하므로,
        # super().load_state_dict(state_dict) 후 param_groups를 다시 연결해주는 것이 안전합니다.
        self.base_optimizer.param_groups = self.param_groups
        # To fully support saving/loading, base_optimizer's state might need explicit handling here
        # if it's not automatically managed by super().load_state_dict or if its state keys
        # are nested within the SAM's state_dict.
        # For now, this re-linking of param_groups is a minimal step.