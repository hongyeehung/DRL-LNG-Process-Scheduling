import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import PPO, SAC, TD3
from tqdm import tqdm
import matplotlib as mpl
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 环境与参数定义 ===
TOTAL_LP = 28
TOTAL_HP = 9
TOTAL_ORV = 5
TOTAL_SCV = 9

LP_CAPACITY = 495 * 0.44
HP_CAPACITY = 460 * 0.44
SCV_CAPACITY = 180

EB_LP = 214  # kW
EB_HP = 1290  # kW
EB_ORV = 900  # kW
QV_SCV = 12.93 * 180 * 24 + 300.9

METRICS = {
    'episodes': [],
    'train_switch_counts': [],
    'schedules': [],
    'runtime_diff': [],
    'close_penalty': [],
    'energy': [],
    'energy_breakdown': [],
    'cum_run_snapshots': [],
    'reward': [],
    'metrics': [],
}

# 负荷与台数计算函数
def compute_loads(Msum, Tsea):
    Mlp = Msum + 156
    Mrec = 12 if 100 < Msum <= 234 else 0
    Mbog = 24
    Mboost = Mbog - Mrec
    orv_unit = 180 if Tsea >= 9 else 0.04225*Tsea**4 - 1.102*Tsea**3 + 7.801*Tsea**2 + 6.445*Tsea + 16.11
    Morv = orv_unit * TOTAL_ORV
    Mscv = Msum - Morv - Mboost - 76
    Mhp = Mlp + Mrec - 150
    n_lp = int(np.ceil(Mlp / LP_CAPACITY))
    n_hp = int(np.ceil(Mhp / HP_CAPACITY))
    n_orv = int(np.ceil(Morv / orv_unit))
    n_scv = int(np.ceil(Mscv / SCV_CAPACITY))
    return n_lp, n_hp, n_orv, n_scv

# 奖励与能耗计算函数
def compute_metrics(cum_run, schedule):
    lp_groups = [range(0, 3), range(3, 6), range(6, 9), range(9, 12),
                 range(12, 16), range(16, 20), range(20, 24), range(24, 28)]
    hp_group = range(TOTAL_LP, TOTAL_LP + TOTAL_HP)
    orv_group = range(TOTAL_LP + TOTAL_HP, TOTAL_LP + TOTAL_HP + TOTAL_ORV)
    scv_group = range(TOTAL_LP + TOTAL_HP + TOTAL_ORV, TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV)
    all_groups = lp_groups + [hp_group, orv_group, scv_group]
    total_diff = 0.0
    for group in all_groups:
        group_runs = cum_run[list(group)]
        if len(group_runs) > 1:
            sorted_runs = np.sort(group_runs)
            diffs = np.diff(sorted_runs)
            total_diff += np.sum(np.abs(diffs - 1200))
    close_penalty = 0.0
    threshold = 100
    for group in all_groups:
        group_runs = cum_run[list(group)]
        if len(group_runs) > 1:
            for i in range(len(group_runs)):
                for j in range(i + 1, len(group_runs)):
                    diff = abs(group_runs[i] - group_runs[j])
                    if diff < threshold:
                        close_penalty += (threshold - diff)
    e_lp = schedule[:TOTAL_LP].sum() * EB_LP * 24
    e_hp = schedule[TOTAL_LP:TOTAL_LP+TOTAL_HP].sum() * EB_HP * 24
    e_orv = schedule[TOTAL_LP+TOTAL_HP:TOTAL_LP+TOTAL_HP+TOTAL_ORV].sum() * EB_ORV * 24
    e_scv = schedule[TOTAL_LP+TOTAL_HP+TOTAL_ORV:TOTAL_LP+TOTAL_HP+TOTAL_ORV+TOTAL_SCV].sum() * \
            ((700+3)*24*0.1229 + QV_SCV*24*1.33)
    energy_total = e_lp + e_hp + e_orv + e_scv
    operational_reward = 0.0
    operational_sections = [
        (0, TOTAL_LP),
        (TOTAL_LP, TOTAL_LP + TOTAL_HP),
        (TOTAL_LP + TOTAL_HP, TOTAL_LP + TOTAL_HP + TOTAL_ORV),
        (TOTAL_LP + TOTAL_HP + TOTAL_ORV, TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV)
    ]
    all_sections_operational = True
    for start, end in operational_sections:
        if end > start and schedule[start:end].sum() == 0:
            all_sections_operational = False
            break
    if all_sections_operational:
        operational_reward += 200.0
    metrics = {
        'std_dev': {},
        'mean_diff': {},
        'std_diff': {}
    }
    group_names = ['lp', 'hp', 'orv', 'scv']
    group_list = [lp_groups, [hp_group], [orv_group], [scv_group]]
    for name, groups in zip(group_names, group_list):
        for i, group in enumerate(groups):
            group_runs = cum_run[list(group)]
            if len(group_runs) > 1:
                sorted_runs = np.sort(group_runs)
                diffs = np.diff(sorted_runs)
                metrics['std_dev'][f'{name}_{i}'] = np.std(group_runs)
                metrics['mean_diff'][f'{name}_{i}'] = np.mean(diffs) if diffs.size > 0 else 0
                metrics['std_diff'][f'{name}_{i}'] = np.std(diffs) if diffs.size > 0 else 0
    
    return total_diff, close_penalty, energy_total, operational_reward, [e_lp, e_hp, e_orv, e_scv], metrics

# 自定义环境类
class LNGEnv(gym.Env):
    def __init__(self):
        super().__init__()
        n = TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV
        self.observation_space = spaces.Box(0, np.inf, shape=(n*5+2,), dtype=np.float32)
        self.action_space = spaces.Box(0, 1, shape=(n,), dtype=np.float32)
        self.current_episode_schedules = []
        self.max_episode_steps = 600
        self.episode_step = 0
        self.last_sched = None
        self.last_total_diff = None  # 新增
        self.switch_counts = np.zeros(n, dtype=int)
        self.reset()
    
    def reset(self):
        # 在一个训练周期结束后，将当前周期的调度序列存储进 METRICS，并重置
        if hasattr(self, 'current_episode_schedules') and self.current_episode_schedules:
            METRICS.setdefault('episodes', []).append(np.array(self.current_episode_schedules))
        self.current_episode_schedules = []
        if hasattr(self, 'switch_counts'):
            METRICS['train_switch_counts'].append(self.switch_counts.copy())

        # 计算设备总数
        n_total_devices = TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV
        
        # ---- 修改点：对 cum_run 进行初始化，使得每个阶段内各设备的累计运行时长差 >2400 且参差不齐 ----
        # 首先申请一个全零数组
        self.cum_run = np.zeros(n_total_devices, dtype=np.float32)
        # cont_run, cum_stop, cont_stop, status 重新置零
        self.cont_run = np.zeros(n_total_devices, dtype=np.float32)
        self.cum_stop = np.zeros(n_total_devices, dtype=np.float32)
        self.cont_stop = np.zeros(n_total_devices, dtype=np.float32)
        self.status = np.zeros(n_total_devices, dtype=int)
        self.last_sched = None
        self.last_total_diff = None  # 重置

        # 对每个阶段内的设备分别单独初始化 cum_run
        # 各阶段设备的索引范围：
        lp_indices  = list(range(0, TOTAL_LP))
        hp_indices  = list(range(TOTAL_LP, TOTAL_LP + TOTAL_HP))
        orv_indices = list(range(TOTAL_LP + TOTAL_HP, TOTAL_LP + TOTAL_HP + TOTAL_ORV))
        scv_indices = list(range(TOTAL_LP + TOTAL_HP + TOTAL_ORV, n_total_devices))
        
        # Helper：对一个小阶段的设备索引列表进行 cum_run 赋值
        def init_stage_cumrun(indices):
            if len(indices) == 0:
                return
            # 为第一个设备设置一个随机基础值（0~1000 之间）
            base = np.random.randint(3000, 6000)
            # 累计值从 base 开始
            cumulative = base
            # 将第一个设备赋值为 base
            self.cum_run[indices[0]] = float(cumulative)
            # 对该阶段后续设备，依次累加 (2400 + 随机偏差)
            for idx in indices[1:]:
                # 随机偏差在 [0, 1000) 之间，可根据需求调节上下限
                extra_offset = np.random.randint(1000, 5000)
                spacing = 2400 + extra_offset
                cumulative += spacing
                self.cum_run[idx] = float(cumulative)
        
        # 分别对 LP、HP、ORV、SCV 阶段的设备进行初始化
        init_stage_cumrun(lp_indices)
        init_stage_cumrun(hp_indices)
        init_stage_cumrun(orv_indices)
        init_stage_cumrun(scv_indices)
        # ---- 修改结束 ----

        # 其他状态维持原样
        self.Msum = np.random.uniform(100, 1260)
        self.Tsea = np.random.uniform(1, 16)
        self.episode_step = 0
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([
            self.status, 
            self.cont_run, 
            self.cont_stop,
            self.cum_run, 
            self.cum_stop, 
            [self.Msum, self.Tsea]
        ]).astype(np.float32)

    def _update_states(self, sched):
        lp_end = TOTAL_LP
        hp_end = TOTAL_LP + TOTAL_HP
        orv_end = TOTAL_LP + TOTAL_HP + TOTAL_ORV
        scv_end = TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV
        max_run = np.zeros(scv_end, int)
        max_run[:lp_end] = 21
        max_run[lp_end:hp_end] = 7
        max_run[hp_end:orv_end] = 90
        max_run[orv_end:scv_end] = 90
        max_stop = 4
        for i, flag in enumerate(sched):
            if self.status[i] == 2:
                self.cont_run[i] = 0
                self.cont_stop[i] += 1
                if self.cont_stop[i] >= max_stop:
                    self.status[i] = 0
                    self.cont_stop[i] = 0
                continue
            if flag:
                self.status[i] = 1
                self.cont_run[i] += 1
                self.cont_stop[i] = 0
                self.cum_run[i] += 24
                if self.cont_run[i] >= max_run[i]:
                    self.status[i] = 2
                    self.cont_run[i] = 0
                    self.cont_stop[i] = 1
            else:
                self.status[i] = 0
                self.cont_stop[i] += 1
                self.cont_run[i] = 0
                self.cum_stop[i] += 24

    def step(self, action):
        self.episode_step += 1
        n_lp, n_hp, n_orv, n_scv = compute_loads(self.Msum, self.Tsea)
        
        # Action decoding and scheduling logic (unchanged)
        sched = np.zeros_like(action, dtype=int)
        idx = np.argsort(-action)
        for cnt, offset in [(n_lp, 0), (n_hp, TOTAL_LP), (n_orv, TOTAL_LP+TOTAL_HP), (n_scv, TOTAL_LP+TOTAL_HP+TOTAL_ORV)]:
            sched[idx[offset:offset+cnt]] = 1

        sections = [
            (0, TOTAL_LP),
            (TOTAL_LP, TOTAL_LP + TOTAL_HP),
            (TOTAL_LP + TOTAL_HP, TOTAL_LP + TOTAL_HP + TOTAL_ORV),
            (TOTAL_LP + TOTAL_HP + TOTAL_ORV, TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV)
        ]
        for start, end in sections:
            if sched[start:end].sum() == 0:
                idx_max = start + np.argmax(action[start:end])
                sched[idx_max] = 1

        penalty = 0
        for i in range(len(sched)):
            if sched[i] == 1 and self.status[i] == 2:
                penalty -= 10000
                sched[i] = 0

        if self.last_sched is not None:
            switch_count = np.sum(sched != self.last_sched)
        else:
            switch_count = 0
        self.last_sched = sched.copy()

        METRICS['schedules'].append(sched.copy())
        self.current_episode_schedules.append(self.status.copy())
        self._update_states(sched)

        # Corrected unpacking with six variables
        total_diff, close_penalty, energy, operational_reward, energy_breakdown, metrics = compute_metrics(self.cum_run, sched)
        
        # Reward calculation (unchanged)
        k1 = 1
        runtime_diff_reward = -k1 * total_diff
        
        k4 = 0.1
        close_penalty_reward = -k4 * close_penalty
        
        k3 = 0.1
        if self.last_total_diff is not None:
            delta_diff = total_diff - self.last_total_diff
            diff_change_reward = -k3 * delta_diff
        else:
            diff_change_reward = 0
        self.last_total_diff = total_diff
        
        k2 = 1
        switch_penalty = -k2 * switch_count
        
        reward = operational_reward + runtime_diff_reward + diff_change_reward + switch_penalty + penalty + close_penalty_reward

        # Record metrics (updated to include metrics)
        METRICS['runtime_diff'].append(total_diff)
        METRICS.setdefault('close_penalty', []).append(close_penalty)
        METRICS['energy'].append(energy)
        METRICS['energy_breakdown'].append(energy_breakdown)
        METRICS.setdefault('cum_run_snapshots', []).append(self.cum_run.copy())
        METRICS.setdefault('reward', []).append(reward)
        METRICS.setdefault('metrics', []).append(metrics)  # Record the metrics dictionary

        done = self.episode_step >= self.max_episode_steps
        return self._get_obs(), reward, done, {}


    def render(self, mode='human'): pass

# 可视化运行时间差异和分布函数
def visualize_runtime_differences(cum_run, schedule):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('各类设备累计运行时间分析', fontsize=16)
    # === 低压泵（LP）组内运行时间绝对差异 ===
    lp_groups = [range(0, 3), range(3, 6), range(6, 9), range(9, 12),
                 range(12, 16), range(16, 20), range(20, 24), range(24, 28)]
    for i, group in enumerate(lp_groups):
        group_runs = cum_run[list(group)]
        if len(group_runs) > 1:
            # 计算相邻设备间的绝对差值
            diffs = [abs(group_runs[j + 1] - group_runs[j]) for j in range(len(group_runs) - 1)]
            axes[0, 0].bar([x + i * 0.1 for x in range(len(diffs))], diffs, width=0.1, label=f'储罐{i+1}', alpha=0.7)
    axes[0, 0].set_title('低压泵组内运行时间绝对差异（按编号顺序）')
    axes[0, 0].set_xlabel('储罐组内设备对')
    axes[0, 0].set_ylabel('运行时间绝对差异 (小时)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # === 高压泵（HP）运行时间绝对差异 ===
    hp_runs = cum_run[TOTAL_LP:TOTAL_LP + TOTAL_HP]
    if len(hp_runs) > 1:
        # 计算相邻设备间的绝对差值
        diffs = [abs(hp_runs[j + 1] - hp_runs[j]) for j in range(len(hp_runs) - 1)]
        axes[0, 1].bar(range(len(diffs)), diffs, color='tab:orange')
    axes[0, 1].set_title('高压泵运行时间绝对差异（按编号顺序）')
    axes[0, 1].set_xlabel('设备i+1 - 设备i')
    axes[0, 1].set_ylabel('运行时间绝对差异 (小时)')
    axes[0, 1].grid(True)
    # === 开架式气化器（ORV）运行时间绝对差异 ===
    orv_runs = cum_run[TOTAL_LP + TOTAL_HP:TOTAL_LP + TOTAL_HP + TOTAL_ORV]
    if len(orv_runs) > 1:
        # 计算相邻设备间的绝对差值
        diffs = [abs(orv_runs[j + 1] - orv_runs[j]) for j in range(len(orv_runs) - 1)]
        axes[1, 0].bar(range(len(diffs)), diffs, color='tab:green')
    axes[1, 0].set_title('开架式气化器运行时间绝对差异（按编号顺序）')
    axes[1, 0].set_xlabel('设备i+1 - 设备i')
    axes[1, 0].set_ylabel('运行时间绝对差异 (小时)')
    axes[1, 0].grid(True)
    # === 浸没式气化器（SCV）运行时间绝对差异 ===
    scv_runs = cum_run[-TOTAL_SCV:]
    if len(scv_runs) > 1:
        # 计算相邻设备间的绝对差值
        diffs = [abs(scv_runs[j + 1] - scv_runs[j]) for j in range(len(scv_runs) - 1)]
        axes[1, 1].bar(range(len(diffs)), diffs, color='tab:red')
    axes[1, 1].set_title('浸没式气化器运行时间绝对差异（按编号顺序）')
    axes[1, 1].set_xlabel('设备i+1 - 设备i')
    axes[1, 1].set_ylabel('运行时间绝对差异 (小时)')
    axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig('runtime_differences.png')
    plt.show()
    # === 累计运行时间分布图（按值排序） ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('各类设备累计运行时间分布（按值排序）', fontsize=16)
    # 低压泵（LP）组内累计运行时间（按值排序）
    for i, group in enumerate(lp_groups):
        group_runs = cum_run[list(group)]
        # 按值排序
        sorted_runs = sorted(group_runs)
        axes[0, 0].plot(range(len(sorted_runs)), sorted_runs, label=f'储罐{i+1}', marker='o')
        for x, y in zip(range(len(sorted_runs)), sorted_runs):
            axes[0, 0].text(x, y, f'{y:.0f}', fontsize=8, ha='center', va='bottom')
    axes[0, 0].set_title('低压泵累计运行时间（按值排序）')
    axes[0, 0].set_xlabel('设备（按运行时间排序）')
    axes[0, 0].set_ylabel('累计运行时间 (小时)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # 高压泵（HP）累计运行时间（按值排序）
    sorted_hp_runs = sorted(hp_runs)
    axes[0, 1].plot(range(len(sorted_hp_runs)), sorted_hp_runs, color='tab:orange', marker='o')
    for x, y in zip(range(len(sorted_hp_runs)), sorted_hp_runs):
        axes[0, 1].text(x, y, f'{y:.0f}', fontsize=8, ha='center', va='bottom')
    axes[0, 1].set_title('高压泵累计运行时间（按值排序）')
    axes[0, 1].set_xlabel('设备（按运行时间排序）')
    axes[0, 1].set_ylabel('累计运行时间 (小时)')
    axes[0, 1].grid(True)
    # 开架式气化器（ORV）累计运行时间（按值排序）
    sorted_orv_runs = sorted(orv_runs)
    axes[1, 0].plot(range(len(sorted_orv_runs)), sorted_orv_runs, color='tab:green', marker='o')
    for x, y in zip(range(len(sorted_orv_runs)), sorted_orv_runs):
        axes[1, 0].text(x, y, f'{y:.0f}', fontsize=8, ha='center', va='bottom')
    axes[1, 0].set_title('开架式气化器累计运行时间（按值排序）')
    axes[1, 0].set_xlabel('设备（按运行时间排序）')
    axes[1, 0].set_ylabel('累计运行时间 (小时)')
    axes[1, 0].grid(True)
    # 浸没式气化器（SCV）累计运行时间（按值排序）
    sorted_scv_runs = sorted(scv_runs)
    axes[1, 1].plot(range(len(sorted_scv_runs)), sorted_scv_runs, color='tab:red', marker='o')
    for x, y in zip(range(len(sorted_scv_runs)), sorted_scv_runs):
        axes[1, 1].text(x, y, f'{y:.0f}', fontsize=8, ha='center', va='bottom')
    axes[1, 1].set_title('浸没式气化器累计运行时间（按值排序）')
    axes[1, 1].set_xlabel('设备（按运行时间排序）')
    axes[1, 1].set_ylabel('累计运行时间 (小时)')
    axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig('cum_runtime_distribution.png')
    plt.show()

if __name__ == '__main__':
    # 训练部分
    env = LNGEnv()
    model = SAC('MlpPolicy', env, verbose=0, device='cpu')
    total_timesteps = 60000

    with tqdm(total=total_timesteps, desc="训练进度") as pbar:
        def progress_callback(locals, globals):
            pbar.update(1)
            return True
        model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    if env.current_episode_schedules:
        METRICS['episodes'].append(np.array(env.current_episode_schedules))
    model.save('sac_lng.zip')

    # 获取最后一个 episode 的步数
    last_episode_steps = len(METRICS['episodes'][-1])
    # 切片获取最后一个 episode 的指标
    last_episode_reward = METRICS['reward'][-last_episode_steps:]
    last_episode_runtime_diff = METRICS['runtime_diff'][-last_episode_steps:]
    last_episode_close_penalty = METRICS['close_penalty'][-last_episode_steps:]
    last_episode_hp_mean_diffs = [m['mean_diff']['hp_0'] for m in METRICS['metrics'][-last_episode_steps:]]
    # 绘制最后一个 episode 的指标变化
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
    # 总奖励变化
    axes[0].plot(range(last_episode_steps), last_episode_reward, label='总奖励')
    axes[0].set_title('最后一个Episode的总奖励变化')
    axes[0].set_xlabel('Episode步数')
    axes[0].set_ylabel('奖励值')
    axes[0].legend()
    axes[0].grid(True)
    # 运行时间差偏差总和
    axes[1].plot(range(last_episode_steps), last_episode_runtime_diff, label='运行时间差偏差总和', color='orange')
    axes[1].set_title('最后一个Episode的运行时间差偏差总和')
    axes[1].set_xlabel('Episode步数')
    axes[1].set_ylabel('偏差总和')
    axes[1].legend()
    axes[1].grid(True)
    # 接近惩罚变化
    axes[2].plot(range(last_episode_steps), last_episode_close_penalty, label='接近惩罚', color='green')
    axes[2].set_title('最后一个Episode的接近惩罚变化')
    axes[2].set_xlabel('Episode步数')
    axes[2].set_ylabel('惩罚值')
    axes[2].legend()
    axes[2].grid(True)
    # 高压泵组内相邻差值平均值
    axes[3].plot(range(last_episode_steps), last_episode_hp_mean_diffs, label='高压泵组内差值平均值', color='red')
    axes[3].set_title('最后一个Episode的高压泵组内相邻差值平均值')
    axes[3].set_xlabel('Episode步数')
    axes[3].set_ylabel('差值平均值 (小时)')
    axes[3].legend()
    axes[3].grid(True)
    plt.tight_layout()
    plt.savefig('last_episode_training_metrics.png')
    plt.show()

    # 评估部分
    model = SAC.load('sac_lng.zip')
    env = LNGEnv()
    env.max_episode_steps = 50
    obs = env.reset()
    episode_data = {'cum_run': [], 'schedules': [], 'status': []}
    val_step=env.max_episode_steps
    for step in range(1,val_step+1):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_data['cum_run'].append(env.cum_run.copy())
        episode_data['schedules'].append(env.current_episode_schedules[-1].copy())
        episode_data['status'].append(env.status.copy())
        if done:
            print(f"评估Episode在第{step+1}步结束")
            break
    else:
        print(f"评估 Episode 在 {val_step} 步后仍未结束，已达上限。")
    
    # 统计分析
    final_cum_run = episode_data['cum_run'][-1]
    lp_groups = [range(0, 3), range(3, 6), range(6, 9), range(9, 12),
                 range(12, 16), range(16, 20), range(20, 24), range(24, 28)]
    hp_group = range(TOTAL_LP, TOTAL_LP + TOTAL_HP)
    orv_group = range(TOTAL_LP + TOTAL_HP, TOTAL_LP + TOTAL_HP + TOTAL_ORV)
    scv_group = range(TOTAL_LP + TOTAL_HP + TOTAL_ORV, TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV)
    all_groups = lp_groups + [hp_group, orv_group, scv_group]
    group_names = ['低压泵', '高压泵', '开架式气化器', '浸没式气化器']
    
    print("\n评估Episode的累计运行时长分布分析：")
    for name, groups in zip(group_names, [lp_groups, [hp_group], [orv_group], [scv_group]]):
        print(f"\n{name}:")
        for i, group in enumerate(groups):
            group_runs = final_cum_run[list(group)]
            if len(group_runs) > 1:
                sorted_runs = np.sort(group_runs)
                diffs = np.diff(sorted_runs)
                print(f"  组{i+1}:")
                print(f"    累计运行时长: {sorted_runs}")
                print(f"    相邻差值: {diffs}")
                print(f"    差值平均值: {np.mean(diffs):.2f} 小时, 标准差: {np.std(diffs):.2f} 小时")

    # 可视化1：累计运行时长变化
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('评估Episode中各类设备累计运行时长变化', fontsize=18)
    plot_params = [
        {'title': '低压压缩机 (LP)', 'indices': range(0, TOTAL_LP), 'ax': axes[0, 0], 'color': 'tab:blue'},
        {'title': '高压压缩机 (HP)', 'indices': range(TOTAL_LP, TOTAL_LP + TOTAL_HP), 'ax': axes[0, 1], 'color': 'tab:orange'},
        {'title': '开架式气化器 (ORV)', 'indices': range(TOTAL_LP + TOTAL_HP, TOTAL_LP + TOTAL_HP + TOTAL_ORV), 'ax': axes[1, 0], 'color': 'tab:green'},
        {'title': '浸没式气化器 (SCV)', 'indices': range(TOTAL_LP + TOTAL_HP + TOTAL_ORV, TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV), 'ax': axes[1, 1], 'color': 'tab:red'}
    ]
    for params in plot_params:
        ax = params['ax']
        for i, device_idx in enumerate(params['indices']):
            device_cum_run = [episode_data['cum_run'][t][device_idx] for t in range(val_step)]
            ax.plot(range(val_step), device_cum_run, color=params['color'], alpha=0.5,
                    label=f'{params["title"].split(" ")[0]}' if i == 0 else None)
        ax.set_title(params['title'])
        ax.grid(True)
        ax.legend(loc='best')
    fig.text(0.5, 0.04, '评估Episode中的步数', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, '累计运行时长 (小时)', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.08, 0.05, 0.95, 0.95])
    plt.savefig('cum_runtime_trends.png')
    plt.show()

    # 可视化2：运行时间差异和分布
    visualize_runtime_differences(episode_data['cum_run'][-1], episode_data['schedules'][-1])

    # 可视化3：设备调度甘特图
    fig, ax = plt.subplots(figsize=(18, 10))
    n_devs = TOTAL_LP + TOTAL_HP + TOTAL_ORV + TOTAL_SCV
    for dev in range(n_devs):
        state_periods = {0: [], 1: [], 2: []}
        start = None
        last_state = None
        for t in range(val_step):
            state = episode_data['status'][t][dev]
            if start is None:
                start = t
                last_state = state
            elif state != last_state or t == val_step-1:
                end = t if state != last_state else t + 1
                state_periods[last_state].append((start, end - start))
                start = t
                last_state = state
        if state_periods[1]:
            color = 'tab:blue' if dev < TOTAL_LP else 'tab:orange' if dev < TOTAL_LP + TOTAL_HP else 'tab:green' if dev < TOTAL_LP + TOTAL_HP + TOTAL_ORV else 'tab:red'
            ax.broken_barh(state_periods[1], (dev, 0.8), facecolors=color)
        if state_periods[2]:
            ax.broken_barh(state_periods[2], (dev, 0.8), facecolors='black')
        if state_periods[0]:
            ax.broken_barh(state_periods[0], (dev, 0.8), facecolors='lightgray', alpha=0.3)
    for pos in [TOTAL_LP, TOTAL_LP + TOTAL_HP, TOTAL_LP + TOTAL_HP + TOTAL_ORV]:
        ax.axhline(y=pos, color='black', linestyle='--', alpha=0.3)
    plt.title('评估Episode的设备调度甘特图')
    plt.xlabel('时间步')
    plt.ylabel('设备编号')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:blue', label='低压压缩机（运行）'),
        Patch(facecolor='tab:orange', label='高压压缩机（运行）'),
        Patch(facecolor='tab:green', label='开架式气化器（运行）'),
        Patch(facecolor='tab:red', label='浸没式气化器（运行）'),
        Patch(facecolor='black', label='强制停机'),
        Patch(facecolor='lightgray', label='未运行')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.savefig('gantt_chart.png')
    plt.show()