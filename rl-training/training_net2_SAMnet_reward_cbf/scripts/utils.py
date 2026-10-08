import gc
import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

#归一化价值函数，用于稳定训练过程中的价值估计
class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    #归一化，真实值->标准化值用于计算loss
    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    # 反归一化，标准化值->真实值用于GAE计算
    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

#构建多层感知机，用于将通用的向量特征转化为一维抽象的特征向量具有高层语义比如现在是否危险，输入下层网络
def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

#实现多变量Beta分布，用于动作空间在有限区间内的连续控制任务，此处代表xyz三个维度独立动作整体动作的beta分布
class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

# Beta分布Actor网络，输出动作的alpha和beta参数
class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

#计算优势函数
class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            #TD_error delta_t = r_t + gamma * V(s_{t+1}) * (1-d_t) - V(s_t)
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            #AGE A_t = delta_t + (gamma * lambda * (1-d_t) * A_{t+1})
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        # 计算回报 G_t = A_t + V(s_t) critic网络的估计值
        returns = advantages + value
        return advantages, returns

#大批量数据切分成小批量数据，使用索引抽出小批量数据
def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) #大批量数据展平
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)#随机打乱并重塑为(num_minibatches, batch_size_per_minibatch)
    for indices in perm:
        yield tensordict[indices]

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int = 0,
    exploration_type: ExplorationType = ExplorationType.MEAN,
):
    """Run metric-only evaluation without retaining full GPU trajectories."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(cfg.device)
    env.enable_render(False)
    env.eval()
    env.set_seed(seed)

    num_envs = env.num_envs
    finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    metric_sums = {}
    completed = 0
    tensordict = env.reset()

    # Evaluation-only trajectory metrics. SPL counts failed episodes as zero,
    # while arrival time is averaged over successful episodes only.
    base_env = env.base_env if hasattr(env, "base_env") else env
    previous_pos = base_env.drone.pos.reshape(num_envs, -1, 3)[:, 0].clone()
    initial_distance = torch.linalg.norm(
        base_env.target_pos.reshape(num_envs, -1, 3)[:, 0] - previous_pos,
        dim=-1,
    )
    path_length = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    spl_sum = 0.0
    success_time_sum = 0.0
    successful_episodes = 0
    sim_dt = float(getattr(base_env, "sim_dt", cfg.sim.dt * cfg.sim.substeps))

    try:
        with set_exploration_type(exploration_type):
            for _ in range(env.max_episode_length):
                tensordict = policy(tensordict)
                transition, next_tensordict = env.step_and_maybe_reset(tensordict)
                next_data = transition["next"]
                current_pos = next_data["info", "drone_state"][..., :3].reshape(
                    num_envs, -1, 3
                )[:, 0]
                active = ~finished
                path_length[active] += torch.linalg.norm(
                    current_pos[active] - previous_pos[active], dim=-1
                )
                previous_pos[active] = current_pos[active]
                done = next_data["done"].reshape(num_envs, -1).any(dim=-1)
                newly_done = done & ~finished

                if newly_done.any():
                    stats = next_data["stats"]
                    for key, value in stats.items():
                        value = value.reshape(num_envs, -1).mean(dim=-1)
                        metric_sums[key] = metric_sums.get(key, 0.0) + value[newly_done].float().sum().item()
                    success = stats["reach_goal"].reshape(num_envs, -1).any(dim=-1)
                    newly_successful = newly_done & success
                    if newly_successful.any():
                        spl = initial_distance[newly_successful] / torch.maximum(
                            path_length[newly_successful],
                            initial_distance[newly_successful],
                        ).clamp_min(1e-6)
                        spl_sum += spl.sum().item()
                        episode_steps = stats["episode_len"].reshape(
                            num_envs, -1
                        ).mean(dim=-1)
                        success_time_sum += (
                            episode_steps[newly_successful].sum().item() * sim_dt
                        )
                        successful_episodes += int(newly_successful.sum().item())
                    completed += int(newly_done.sum().item())
                    finished |= newly_done

                del transition, next_data
                tensordict = next_tensordict
                if finished.all():
                    break

        if completed == 0:
            raise RuntimeError("evaluation completed without a finished episode")

        info = {
            "eval/stats." + key: total / completed
            for key, total in metric_sums.items()
        }
        info["eval/stats.spl"] = spl_sum / completed
        info["eval/stats.success_time"] = (
            success_time_sum / successful_episodes
            if successful_episodes > 0
            else float("nan")
        )
        if torch.cuda.is_available():
            info["eval/system.gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(cfg.device) / (1024 ** 3)
            info["eval/system.gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(cfg.device) / (1024 ** 3)
    finally:
        env.enable_render(not cfg.headless)
        env.train()
        env.reset()
        gc.collect()
        torch.cuda.empty_cache()

    return info


def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coordinate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"


