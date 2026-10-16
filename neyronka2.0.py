import torch #для работы с многомерными массивами
import torch.nn as nn #для построения нейронной сети
import torch.optim as optim #для оптимизации
import random
import numpy as np
try:
    import pygame
except ImportError:
    pygame = None  # Без окна можно выполнять домашку без pygame.
import math

from collections import deque #для памяти

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_INPUTS = 9
N_ACTIONS = 2
HIDDEN = 64

LR = 0.0003
GAMMA = 0.97
BATCH_SIZE = 128
MEMORY_SIZE = 100_000


MAX_SPEED = 10

STUCK_SPEED = 0.05
MAX_STUCK_STEPS = 120

EVAL_EPISODES = 20
EVAL_MAX_STEPS = 3000

RENDER = False
DEMO_MAX_STEPS = 30

# Урок 2: стартовые настройки, пока без подбора гиперпараметров.
ACTOR_LR = LR
CRITIC_LR = LR
TAU = 0.005
ACTION_NOISE = 0.15
START_STEPS = 1000
TRAIN_MAX_STEPS = 1000 #1000(1), 1500(2), 900(3)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUTS, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
            nn.Tanh()  #сжимает от -1 до 1
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUTS + N_ACTIONS, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size) #выбираем определённое количество случайных элементов
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)), #тензор из float32
            torch.FloatTensor(np.array(a)),  #R1
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


WIDTH, HEIGHT = 900, 700

LEVEL = 3
if LEVEL == 1:
    walls = [[(50, 100), (850, 100)], [(50, 250), (700, 250)],
             [(850, 100), (850, 600)], [(700, 250), (700, 600)]]
    CHECKPOINTS = [(400, 175), (775, 300)]
    ST_X, ST_Y = 60, 175
elif LEVEL == 2:
    walls = [[(50, 50), (850, 50)], [(850, 50), (850, 650)],
             [(850, 650), (50, 650)], [(50, 650), (50, 50)],
             [(200, 200), (700, 200)], [(700, 200), (700, 500)],
             [(700, 500), (200, 500)], [(200, 500), (200, 200)]]
    CHECKPOINTS = [(450, 575), (775, 350), (450, 125), (125, 350)]
    ST_X, ST_Y = 150, 550
else:
    walls = [[(50, 50), (270, 50)], [(150, 150), (170, 150)],
             [(170, 150), (170, 400)], [(170, 500), (170, 650)],
             [(270, 50), (270, 400)], [(270, 500), (270, 550)],
             [(170, 650), (850, 650)], [(270, 550), (750, 550)],
             [(750, 550), (750, 150)], [(850, 650), (850, 50)],
             [(850, 50), (400, 50)], [(750, 150), (500, 150)],
             [(500, 150), (500, 500)], [(400, 50), (400, 400)],
             [(400, 400), (270, 400)], [(170, 400), (150, 400)],
             [(500, 500), (270, 500)], [(170, 500), (50, 500)],
             [(50, 500), (50, 50)], [(150, 400), (150, 150)]]
    CHECKPOINTS = [(220, 175), (220, 550), (700, 600), (800, 200), (450, 200), (100, 450)]
    ST_X, ST_Y = 100, 100

screen = myfont = clock = None
if RENDER:
    if pygame is None:
        raise ImportError("Установите pygame или поставьте RENDER = False")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Actor-Critic Car: homework")
    myfont = pygame.font.SysFont('Arial', 18)
    clock = pygame.time.Clock()
MAX_RAY_LEN = 200

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (250, 0, 0)
GREEN = (0, 250, 0)


def update_speed(speed, drive):
    speed += drive * 0.1
    return max(0.0, min(speed, MAX_SPEED))


def target_features(x, y, angle, target):
    """Дистанция, sin и cos направления относительно курса автомобиля."""
    cx, cy = target
    dx = cx - x
    dy = y - cy  # На экране y растёт вниз; здесь переводим ось вверх.
    distance = math.hypot(dx, dy)
    if distance < 1e-9:
        # В самой точке направление не определено; выбираем условное значение.
        return [0.0, 0.0, 1.0]
    target_angle = math.atan2(dy, dx)
    relative_angle = target_angle - math.radians(angle)
    return [
        distance / math.hypot(WIDTH, HEIGHT),
        math.sin(relative_angle),
        math.cos(relative_angle),
    ]


class Car:
    def __init__(self, x, y):
        self.start_x = x
        self.start_y = y
        self.max_speed = MAX_SPEED
        self.rotation_speed = 3
        self.width = 30
        self.height = 15
        self.surface = None
        if RENDER:
            self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.surface.fill(RED)
        self.reset()

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.speed = 2.0
        self.angle = 0
        self.alive = True
        self.reward = 0
        self.unreward = 0
        self.break_t = 0
        self.norm_distance = [1.0] * 5
        self.distance = [MAX_RAY_LEN] * 5
        self.current_checkpoint_idx = 0
        # ПРАВКА 8: счётчик "застревания"
        self.stuck_steps = 0
        # ПРАВКА 9: счётчики для честного отчёта метрики на уроке
        self.checkpoints_passed = 0
        self.collisions = 0
        # ПРАВКА 10: для baseline-метрик (сравнение моделей между собой)
        self.total_progress = 0.0
        self.speed_sum = 0.0
        self.steps_alive = 0
        self.max_speed_reached = 0.0

    def drive(self):
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y -= math.sin(rad) * self.speed

    def get_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0: return None
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))
        return None

    def cast_rays(self, walls):
        ray_angles = [self.angle - 60, self.angle - 30, self.angle,
                      self.angle + 30, self.angle + 60]
        self.norm_distance = []
        self.distance = []
        for ra in ray_angles:
            rad = math.radians(ra)
            dx = self.x + math.cos(rad) * MAX_RAY_LEN
            dy = self.y - math.sin(rad) * MAX_RAY_LEN
            closest = MAX_RAY_LEN
            for wall in walls:
                pt = self.get_intersection((self.x, self.y), (dx, dy), wall[0], wall[1])
                if pt:
                    d = math.hypot(pt[0] - self.x, pt[1] - self.y)
                    if d < closest: closest = d
            self.norm_distance.append(closest / MAX_RAY_LEN)
            self.distance.append(closest)
            if RENDER:
                color = GREEN if closest > 50 else RED
                pygame.draw.line(screen, color, (self.x, self.y),
                                  (self.x + math.cos(rad) * closest,
                                   self.y - math.sin(rad) * closest))

    def check_collision(self, walls):
        rad = math.radians(self.angle)
        cos_a = math.cos(rad); sin_a = math.sin(rad)
        w = self.width / 2; h = self.height / 2
        corners = [(w, h), (w, -h), (-w, -h), (-w, h)]
        world = [(self.x + lx * cos_a + ly * sin_a,
                  self.y - lx * sin_a + ly * cos_a) for lx, ly in corners]
        sides = [(world[i], world[(i + 1) % 4]) for i in range(4)]
        for side in sides:
            for wall in walls:
                if self.get_intersection(side[0], side[1], wall[0], wall[1]):
                    self.alive = False
                    self.collisions += 1
                    return True
        if not (0 < self.x < WIDTH and 0 < self.y < HEIGHT):
            self.alive = False
            self.collisions += 1
            return True
        return False

    def get_checkpoint_distance(self):
        cx, cy = CHECKPOINTS[self.current_checkpoint_idx]
        return math.hypot(cx - self.x, cy - self.y)

    def get_checkpoint_angle(self):
        cx, cy = CHECKPOINTS[self.current_checkpoint_idx]
        dx = cx - self.x
        dy = self.y - cy
        target_angle = math.degrees(math.atan2(dy, dx))
        diff = target_angle - self.angle
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        return diff / 180.0

    def check_checkpoint(self):
        if self.get_checkpoint_distance() < 55:
            self.current_checkpoint_idx = (self.current_checkpoint_idx + 1) % len(CHECKPOINTS)
            self.checkpoints_passed += 1
            return 35.0
        return 0.0

    def apply_action(self, action):
        # Функция урока: непрерывный руль и газ/тормоз.
        if not self.alive:
            return
        steer, drive = action
        steer = max(-1.0, min(float(steer), 1.0))
        drive = max(-1.0, min(float(drive), 1.0))
        self.speed = update_speed(self.speed, drive)
        self.angle = (self.angle - steer * self.rotation_speed) % 360.0
        # Сохраняем смысл break_t для нашей прежней функции награды.
        self.break_t = self.break_t + 1 if drive < 0 else 0

    def update_metrics(self):
        self.steps_alive += 1
        self.speed_sum += self.speed
        self.max_speed_reached = max(self.max_speed_reached, self.speed)
        if self.speed < STUCK_SPEED:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

    def draw(self):
        if not RENDER:
            return
        rotated = pygame.transform.rotate(self.surface, self.angle)
        screen.blit(rotated, rotated.get_rect(center=(self.x, self.y)))


class DDPGAgent:
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.target_actor = Actor()
        self.target_critic = Critic()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.eval()
        self.target_critic.eval()
        for network in (self.target_actor, self.target_critic):
            for parameter in network.parameters():
                parameter.requires_grad_(False)

        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps = 0
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def get_obs(self, car):
        target = CHECKPOINTS[car.current_checkpoint_idx]
        return car.norm_distance + [car.speed / MAX_SPEED] + target_features(
            car.x, car.y, car.angle, target
        )

    def choose_action(self, obs, noise_std=0.0):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)  #B1
            action = self.actor(t)  #B2
            action = action.squeeze(0).numpy()
        if noise_std > 0:
            action = action + np.random.normal(0.0, noise_std, size=N_ACTIONS)
        return np.clip(action, -1.0, 1.0).tolist()

    def remember(self, obs, action, reward, next_obs, done):
        self.memory.push(obs, action, reward, next_obs, done)

    def save(self, path="ddpg_weights.pth"):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)
        print(f"Модель сохранена {path}")

    def load(self, path="ddpg_weights.pth"):
        weights = torch.load(path, map_location="cpu", weights_only=True)
        self.actor.load_state_dict(weights["actor"])
        self.critic.load_state_dict(weights["critic"])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        print(f"Модель загружена {path}")


    # критик получает цель r + gamma * Q_next.
    @torch.no_grad()
    def compute_target(self, r, s2, d):
        # r и d должны быть столбцами [B, 1]. d=1 только при аварии.
        next_action = self.target_actor(s2)
        q_next = self.target_critic(s2, next_action)
        return r + GAMMA * (1.0 - d) * q_next

    def critic_step(self, s, a, target):
        # Оцениваем именно действие из памяти.
        q = self.critic(s, a)
        assert q.shape == target.shape, "Оба тензора должны иметь форму [B, 1]"
        loss = nn.SmoothL1Loss()(q, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        return loss.item(), q.detach().mean().item()

    # УРОК 2, блок 2: актор выбирает действия с высокой оценкой критика.
    def actor_step(self, s):
        # Веса критика фиксированы, но производная по его входу action нужна.
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        try:
            actions = self.actor(s)
            loss = -self.critic(s, actions).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
        return loss.item()

    # УРОК 2, блок 3: целевые копии медленно догоняют обучаемые сети.
    @torch.no_grad()
    def soft_update(self):
        for online, target in [(self.actor, self.target_actor),
                               (self.critic, self.target_critic)]:
            for parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.mul_(1.0 - TAU)
                target_parameter.add_(parameter, alpha=TAU)

    def learn(self, update_actor=True):
        if len(self.memory) < BATCH_SIZE:
            return None
        s, a, r, s2, d = self.memory.sample(BATCH_SIZE)
        r = r.unsqueeze(1)  # [B] -> [B, 1]
        d = d.unsqueeze(1)  # иначе broadcasting может создать [B, B]
        target = self.compute_target(r, s2, d)
        critic_loss, q_mean = self.critic_step(s, a, target)
        actor_loss = self.actor_step(s) if update_actor else None
        self.soft_update()
        self.steps += 1
        return {"critic_loss": critic_loss, "actor_loss": actor_loss,
                "q_mean": q_mean, "target_mean": target.mean().item()}


def compute_reward(car, old_dist, new_dist):
    r = 1.1 * car.speed / MAX_SPEED
    r += car.check_checkpoint()

    progress = old_dist - new_dist
    r += max(-2.0, min(2.0, progress * 0.1))

    min_dist = min(car.distance) if car.distance else MAX_RAY_LEN
    if min_dist < 35:
        pen = (35 - min_dist) * 0.15; r -= pen; car.unreward += pen
    if car.speed == 0:
        r -= 1
    car.reward += r
    return r


#Проверка перед заездом
assert math.isclose(update_speed(5, 1), 5.1)
assert update_speed(10, 1) == 10
assert update_speed(0.03, -1) == 0
test_car = Car(ST_X, ST_Y)
test_car.speed = MAX_SPEED
test_car.apply_action((1, 1))
assert test_car.speed == MAX_SPEED and test_car.angle == 357
test_car.cast_rays(walls)

agent = DDPGAgent()
obs = agent.get_obs(test_car)
assert len(obs) == N_INPUTS == 9
s = torch.FloatTensor([obs])
with torch.no_grad():
    a = agent.actor(s)
    q = agent.critic(s, a)
assert a.shape == (1, 2) and q.shape == (1, 1)
assert torch.isfinite(a).all() and (a.abs() <= 1).all()
assert torch.isfinite(q).all()
assert isinstance(agent.actor.net[-1], nn.Tanh)
assert isinstance(agent.critic.net[-1], nn.Linear)
for network, sizes in [(agent.actor, [(9, 64), (64, 64), (64, 2)]),
                       (agent.critic, [(11, 64), (64, 64), (64, 1)])]:
    assert [(x.in_features, x.out_features) for x in network.net
            if isinstance(x, nn.Linear)] == sizes

chosen = agent.choose_action(obs)
assert isinstance(chosen, list) and len(chosen) == 2
assert np.allclose(chosen, a.squeeze(0).numpy())

test_memory = ReplayBuffer(10)
test_memory.push(obs, [0.25, -0.75], 1.0, obs, False)
bs, ba, br, bs2, bd = test_memory.sample(1)
assert ba.dtype == torch.float32 and ba.shape == (1, 2)
assert torch.allclose(ba, torch.tensor([[0.25, -0.75]])), "Нельзя обрезать дробное действие до целого"
print("OK: функции урока, сети, выбор действия и память.")

batch_s = torch.randn(4, N_INPUTS)
batch_a = agent.actor(batch_s)
batch_q = agent.critic(batch_s, batch_a)
assert batch_a.shape == (4, 2) and batch_q.shape == (4, 1)
seen = []
hook = agent.critic.net[0].register_forward_pre_hook(
    lambda module, inputs: seen.append(inputs[0].detach().clone())
)
try:
    agent.critic(batch_s, batch_a)
finally:
    hook.remove()
assert torch.allclose(seen[0][:, :N_INPUTS], batch_s)
assert torch.allclose(seen[0][:, N_INPUTS:], batch_a)

(-batch_q.mean()).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in agent.actor.parameters())
agent.actor.zero_grad(set_to_none=True)
agent.critic.zero_grad(set_to_none=True)
for original, target in [(agent.actor, agent.target_actor), (agent.critic, agent.target_critic)]:
    for p, tp in zip(original.parameters(), target.parameters()):
        assert torch.equal(p, tp) and p.data_ptr() != tp.data_ptr()
print("OK: критик получает действие, градиенты проходят, целевые сети независимы.")


car = Car(ST_X, ST_Y)
agent = DDPGAgent()
ep_reward = 0.0
history = []
terminated = truncated = False

for step in range(DEMO_MAX_STEPS):
    if RENDER:
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            print("Окно закрыто, демонстрация остановлена пользователем.")
            break
        screen.fill(WHITE)
        for wall in walls:
            pygame.draw.line(screen, BLACK, wall[0], wall[1])
        for idx, cp in enumerate(CHECKPOINTS):
            is_active = idx == car.current_checkpoint_idx
            pygame.draw.circle(screen, GREEN if is_active else (180, 180, 180), cp, 12)
            if is_active:
                pygame.draw.circle(screen, GREEN, cp, 55, 1)

    car.cast_rays(walls)
    obs = agent.get_obs(car)
    action = agent.choose_action(obs)  #D1
    old_dist = car.get_checkpoint_distance()

    car.apply_action(action)
    car.drive()
    car.cast_rays(walls)
    car.check_collision(walls)

    if car.alive:
        new_dist = car.get_checkpoint_distance()
        car.total_progress += old_dist - new_dist
        step_reward = compute_reward(car, old_dist, new_dist)
        car.update_metrics()
    else:
        step_reward = -350.0

    # После compute_reward: наблюдение уже указывает на новый чекпоинт.
    next_obs = agent.get_obs(car)
    terminated = not car.alive
    truncated = (not terminated) and (
        car.stuck_steps >= MAX_STUCK_STEPS or step + 1 >= DEMO_MAX_STEPS
    )
    ep_reward += step_reward
    agent.remember(obs, action, step_reward, next_obs, terminated)
    history.append((step + 1, *action, car.speed, step_reward))

    if RENDER:
        car.draw()
        screen.blit(myfont.render(f"Step:{step + 1} R:{ep_reward:.1f}", True, BLACK), (5, 5))
        screen.blit(myfont.render(f"steer:{action[0]:+.2f} drive:{action[1]:+.2f}", True, BLACK), (5, 25))
        pygame.display.flip()
        clock.tick(60)
    if terminated or truncated:
        break

print("Шаг, руль, газ/тормоз, скорость, награда:")
for row in history[::5]:
    print(tuple(round(x, 3) for x in row))
print("Переходов в памяти:", len(agent.memory))
print("Чекпоинтов:", car.checkpoints_passed, "Награда:", round(ep_reward, 2))
print("Авария:", terminated, "Внешняя остановка:", truncated)
if RENDER:
    pygame.quit()


def evaluate_agent(agent, num_episodes=EVAL_EPISODES, max_steps=EVAL_MAX_STEPS, save_path=None):
    eval_car = Car(ST_X, ST_Y)

    rewards, checkpoints, progresses = [], [], []
    avg_speeds, steps_list, collisions, completed = [], [], [], []

    for _ in range(num_episodes):
        eval_car.reset()
        ep_r = 0.0

        for _ in range(max_steps):
            eval_car.cast_rays(walls)
            obs = agent.get_obs(eval_car)
            action = agent.choose_action(obs)

            old_d = eval_car.get_checkpoint_distance()
            eval_car.apply_action(action)
            eval_car.drive()
            eval_car.cast_rays(walls)
            eval_car.check_collision(walls)

            if not eval_car.alive:
                ep_r += -350.0
                break

            new_d = eval_car.get_checkpoint_distance()
            eval_car.total_progress += (old_d - new_d)
            ep_r += compute_reward(eval_car, old_d, new_d)
            eval_car.update_metrics()

            if eval_car.stuck_steps >= MAX_STUCK_STEPS:
                break

        avg_speed = eval_car.speed_sum / eval_car.steps_alive if eval_car.steps_alive > 0 else 0.0

        rewards.append(ep_r)
        checkpoints.append(eval_car.checkpoints_passed)
        progresses.append(eval_car.total_progress)
        avg_speeds.append(avg_speed)
        steps_list.append(eval_car.steps_alive)
        collisions.append(eval_car.collisions)
        completed.append(eval_car.checkpoints_passed >= len(CHECKPOINTS))

    metrics = {
        "avg_reward": float(np.mean(rewards)),
        "best_reward": float(np.max(rewards)),
        "avg_checkpoints": float(np.mean(checkpoints)),
        "max_checkpoints": int(np.max(checkpoints)),
        "avg_progress": float(np.mean(progresses)),
        "avg_speed": float(np.mean(avg_speeds)),
        "avg_steps": float(np.mean(steps_list)),
        "collision_rate": float(np.mean(collisions)),
        "lap_completion_rate": float(np.mean(completed)),
    }

    print("\n========== BASELINE / EVALUATION ==========")
    print(f"Episodes:              {num_episodes}")
    print(f"Average reward:        {metrics['avg_reward']:.2f}")
    print(f"Best reward:           {metrics['best_reward']:.2f}")
    print(f"Average checkpoints:   {metrics['avg_checkpoints']:.2f}")
    print(f"Max checkpoints:       {metrics['max_checkpoints']}")
    print(f"Average progress:      {metrics['avg_progress']:.2f}")
    print(f"Average speed:         {metrics['avg_speed']:.2f}")
    print(f"Average steps:         {metrics['avg_steps']:.1f}")
    print(f"Collision rate:        {metrics['collision_rate']:.2f}")
    print(f"Lap completion rate:   {100.0 * metrics['lap_completion_rate']:.1f}%")
    print("============================================\n")

    # сохраняем в файл, чтобы потом сравнить несколько моделей между собой
    if save_path is not None:
        np.savez(save_path, **metrics)
    return metrics


# Прежняя функция cast_rays рисует только при RENDER.
# Для оценки временно отключаем рисование: окно могло уже закрыться.
previous_render = RENDER
RENDER = False
try:
    metrics = evaluate_agent(agent, num_episodes=2, max_steps=30)
finally:
    RENDER = previous_render
assert all(math.isfinite(value) for value in metrics.values())
print("OK: наш evaluate_agent работает с новой архитектурой.")

r_demo = torch.tensor([[2.0], [2.0]])
d_demo = torch.tensor([[0.0], [1.0]])
q_next_demo = torch.tensor([[10.0], [10.0]])
y_demo = r_demo + GAMMA * (1 - d_demo) * q_next_demo
assert torch.allclose(y_demo, torch.tensor([[11.7], [2.0]]))
print("Продолжение:", y_demo[0].item(), "Авария:", y_demo[1].item())

# Почему всё делаем столбцами [B, 1]?
r_bad = torch.tensor([2.0, 3.0])
print("Ошибка формы: [2] + [2, 1] ->", tuple((r_bad + q_next_demo).shape))
print("Верная форма:", tuple((r_bad.unsqueeze(1) + q_next_demo).shape))

def reset_car(car, agent):
    car.reset()
    car.cast_rays(walls)
    return agent.get_obs(car)


def take_step(car, agent, action, episode_step, max_steps):
    old_dist = car.get_checkpoint_distance()
    car.apply_action(action)
    car.drive()
    car.cast_rays(walls)
    car.check_collision(walls)
    if car.alive:
        new_dist = car.get_checkpoint_distance()
        car.total_progress += old_dist - new_dist
        reward = compute_reward(car, old_dist, new_dist)
        car.update_metrics()
    else:
        reward = -350.0
    next_obs = agent.get_obs(car)
    terminated = not car.alive
    truncated = (not terminated) and (
        car.stuck_steps >= MAX_STUCK_STEPS or episode_step >= max_steps
    )
    return next_obs, reward, terminated, truncated


def collect_random_experience(agent, count=256):
    global RENDER
    previous_render = RENDER
    RENDER = False
    try:
        car = Car(ST_X, ST_Y)
        obs = reset_car(car, agent)
        episode_step = 0
        for _ in range(count):
            action = np.random.uniform(-1.0, 1.0, size=N_ACTIONS).tolist()
            episode_step += 1
            next_obs, reward, terminated, truncated = take_step(
                car, agent, action, episode_step, TRAIN_MAX_STEPS
            )
            agent.remember(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if terminated or truncated:
                obs = reset_car(car, agent)
                episode_step = 0
    finally:
        RENDER = previous_render


# Отдельный агент для опытов урока: не портим состояние агента из домашки.
lesson_agent = DDPGAgent()
collect_random_experience(lesson_agent, count=2 * BATCH_SIZE)
s, a, r, s2, d = lesson_agent.memory.sample(BATCH_SIZE)
r = r.unsqueeze(1)
d = d.unsqueeze(1)
assert s.shape == s2.shape == (BATCH_SIZE, N_INPUTS)
assert a.shape == (BATCH_SIZE, N_ACTIONS)
assert r.shape == d.shape == (BATCH_SIZE, 1)
print("Переходов:", len(lesson_agent.memory))
print("s, a, r, s2, d:", s.shape, a.shape, r.shape, s2.shape, d.shape)

def snapshot(network):
    return [p.detach().clone() for p in network.parameters()] #копия без возможности изменять градиенты, убирает граф


def unchanged(before, network):
    return all(torch.equal(old, new.detach())
               for old, new in zip(before, network.parameters()))


target = lesson_agent.compute_target(r, s2, d)
assert target.shape == (BATCH_SIZE, 1) and not target.requires_grad
actor_before = snapshot(lesson_agent.actor)
critic_before = snapshot(lesson_agent.critic)
target_actor_before = snapshot(lesson_agent.target_actor)
target_critic_before = snapshot(lesson_agent.target_critic)

critic_loss, q_mean = lesson_agent.critic_step(s, a, target)
assert not unchanged(critic_before, lesson_agent.critic)
assert unchanged(actor_before, lesson_agent.actor)
assert unchanged(target_actor_before, lesson_agent.target_actor)
assert unchanged(target_critic_before, lesson_agent.target_critic)
assert math.isfinite(critic_loss)
print("OK: изменился только критик. Loss:", critic_loss)

with torch.no_grad():
    fixed_loss_before = nn.SmoothL1Loss()(lesson_agent.critic(s, a), target).item()
for _ in range(30):
    lesson_agent.critic_step(s, a, target)
with torch.no_grad():
    fixed_loss_after = nn.SmoothL1Loss()(lesson_agent.critic(s, a), target).item()
print(f"Ошибка на фиксированной пачке: {fixed_loss_before:.4f} -> {fixed_loss_after:.4f}")
print("Если ошибка не снижается, разберите формы, detach и шаг оптимизатора.")

actor_before = snapshot(lesson_agent.actor)
critic_before = snapshot(lesson_agent.critic)
actor_loss = lesson_agent.actor_step(s)
assert not unchanged(actor_before, lesson_agent.actor)
assert unchanged(critic_before, lesson_agent.critic)
assert all(p.requires_grad for p in lesson_agent.critic.parameters())
assert math.isfinite(actor_loss)
print("OK: изменился только актор. Actor loss:", actor_loss)

before = snapshot(lesson_agent.target_critic)
online = snapshot(lesson_agent.critic)
lesson_agent.soft_update()
for old, current, actual in zip(before, online, lesson_agent.target_critic.parameters()):
    assert torch.allclose(actual, (1 - TAU) * old + TAU * current)
assert all(p.grad is None for p in lesson_agent.target_actor.parameters())
assert all(p.grad is None for p in lesson_agent.target_critic.parameters())

stats = lesson_agent.learn()
assert stats is not None and all(math.isfinite(v) for v in stats.values())
print("OK: learn выполнил полный шаг DDPG:", stats)


#Исследование и ограниченный запуск
def evaluate_for_checkpoint(agent, max_steps=EVAL_MAX_STEPS):
    """Отдельный заезд без шума, обучения и изменения состояния учебного заезда."""
    global RENDER
    if max_steps <= 0:
        raise ValueError("max_steps должен быть положительным")
    previous_render = RENDER
    previous_mode = agent.actor.training
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    RENDER = False
    try:
        agent.actor.eval()
        eval_car = Car(ST_X, ST_Y)
        obs = reset_car(eval_car, agent)
        reward_sum = 0.0
        steps_to_last_checkpoint = None
        with torch.no_grad():
            for step in range(1, max_steps + 1):
                action = agent.choose_action(obs, noise_std=0.0)
                before = eval_car.checkpoints_passed
                obs, reward, terminated, truncated = take_step(
                    eval_car, agent, action, step, max_steps
                )
                reward_sum += reward
                if eval_car.checkpoints_passed > before:
                    steps_to_last_checkpoint = step
                if terminated or truncated:
                    break
        return {"checkpoints": eval_car.checkpoints_passed,
                "collision": bool(terminated), "steps": step,
                "steps_to_last_checkpoint": steps_to_last_checkpoint,
                "reward": float(reward_sum), "truncated": bool(truncated)}
    finally:
        agent.actor.train(previous_mode)
        RENDER = previous_render
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def checkpoint_rank(metrics):
    # Не поощряем раннюю аварию меньшей общей длиной заезда.
    time_to_checkpoint = metrics["steps_to_last_checkpoint"]
    return (metrics["checkpoints"], not metrics["collision"],
            -time_to_checkpoint if time_to_checkpoint is not None else 0)


def maybe_save_best(agent, eval_every=5000, eval_max_steps=EVAL_MAX_STEPS):
    """Вызывается после шага обучения; на границе интервала оценивает и сохраняет.

    Счётчик training_env_steps общий для train_ddpg и run_pygame этого агента.
    Отдельный каталог на каждый объект агента защищает результаты прошлых запусков.
    """
    from pathlib import Path
    from datetime import datetime
    import csv
    import json
    import uuid

    if eval_every <= 0 or eval_max_steps <= 0:
        raise ValueError("Интервалы оценки должны быть положительными")
    step = getattr(agent, "training_env_steps", 0)
    if step == 0 or step % eval_every:
        return None
    state = getattr(agent, "best_checkpoint_state", None)
    if state is not None:
        if state["eval_max_steps"] != eval_max_steps or state["level"] != LEVEL:
            raise ValueError("Условия оценки изменились. Начните новый эксперимент с новым агентом.")
        if state["last_eval_step"] == step:
            return None

    metrics = evaluate_for_checkpoint(agent, max_steps=eval_max_steps)
    if state is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = Path("ddpg_runs") / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {"run_dir": str(run_dir.resolve()), "best_score": None,
                 "best_step": None, "last_eval_step": None,
                 "eval_max_steps": eval_max_steps, "level": LEVEL, "history": []}
        agent.best_checkpoint_state = state
        print("Каталог результатов:", state["run_dir"])

    run_dir = Path(state["run_dir"])
    score = checkpoint_rank(metrics)
    improved = state["best_score"] is None or score > state["best_score"]
    row = {"training_step": step, **metrics, "is_best": improved}
    if improved:
        # Формат совместим с существующим agent.load(). Это веса, не снимок оптимизаторов.
        checkpoint = {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
                      "training_step": step, "metrics": metrics, "level": LEVEL,
                      "eval_max_steps": eval_max_steps}
        temp_path = run_dir / "best_weights.tmp"
        torch.save(checkpoint, temp_path)
        temp_path.replace(run_dir / "best_weights.pth")
        (run_dir / "best_metrics.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state["best_score"], state["best_step"] = score, step
    csv_path = run_dir / "evaluation_history.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    state["last_eval_step"] = step
    state["history"].append(row)
    print(f"[EVAL {step}] CP:{metrics['checkpoints']} collision:{metrics['collision']} "
          f"steps:{metrics['steps']} reward:{metrics['reward']:.2f} "
          + ("BEST — веса сохранены" if improved else f"best_step:{state['best_step']}"))
    return row


def train_ddpg(agent, total_steps=5000, start_steps=START_STEPS,
               max_episode_steps=TRAIN_MAX_STEPS, noise_std=ACTION_NOISE,
               eval_every=5000, eval_max_steps=EVAL_MAX_STEPS):
    global RENDER
    previous_render = RENDER
    RENDER = False
    episodes, updates = [], []
    try:
        car = Car(ST_X, ST_Y)
        obs = reset_car(car, agent)
        episode_step = 0
        episode_reward = 0.0
        for global_step in range(total_steps):
            if global_step < start_steps:
                action = np.random.uniform(-1, 1, size=N_ACTIONS).tolist()
            else:
                action = agent.choose_action(obs, noise_std=noise_std)
            episode_step += 1
            next_obs, reward, terminated, truncated = take_step(
                car, agent, action, episode_step, max_episode_steps
            )
            agent.remember(obs, action, reward, next_obs, terminated)
            if global_step >= start_steps:
                stats = agent.learn()
                if stats is not None:
                    updates.append({"step": global_step + 1, **stats})
            episode_reward += reward
            obs = next_obs
            agent.training_env_steps = getattr(agent, "training_env_steps", 0) + 1
            maybe_save_best(agent, eval_every=eval_every, eval_max_steps=eval_max_steps)
            budget_end = global_step + 1 == total_steps
            if terminated or truncated or budget_end:
                episodes.append({
                    "step": global_step + 1, "length": episode_step,
                    "reward": episode_reward, "checkpoints": car.checkpoints_passed,
                    "collision": terminated,
                    "truncated": truncated,
                    "partial": budget_end and not (terminated or truncated),
                })
                if len(episodes) % 10 == 0 or budget_end:
                    last = updates[-1] if updates else None
                    print("Step:", global_step + 1, "Episodes:", len(episodes),
                          "Last episode:", episodes[-1], "Losses:", last)
                if not budget_end:
                    obs = reset_car(car, agent)
                    episode_step, episode_reward = 0, 0.0
        return {"episodes": episodes, "updates": updates,
                "evaluations": list(getattr(agent, "best_checkpoint_state", {}).get("history", []))}
    finally:
        RENDER = previous_render


def evaluate_without_window(agent, max_steps=EVAL_MAX_STEPS):
    global RENDER
    previous_render = RENDER
    RENDER = False
    try:
        return evaluate_agent(agent, num_episodes=1, max_steps=max_steps)
    finally:
        RENDER = previous_render

RUN_TRAINING = True
if RUN_TRAINING:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    training_agent = DDPGAgent()
    print("До обучения (без шума):")
    before_metrics = evaluate_without_window(training_agent)
    training_log = train_ddpg(training_agent, total_steps=100000)
    print("После обучения (без шума):")
    after_metrics = evaluate_without_window(training_agent)
    for name in ("avg_reward", "avg_checkpoints", "avg_progress", "collision_rate"):
        print(name, ":", before_metrics[name], "->", after_metrics[name])
else:
    print("Обучение выключено. После разбора поставьте RUN_TRAINING = True.")

def draw_hud(episod, noise_std, ep_reward, loss, car):
    # Те же координаты HUD, что в нашем q-network.ipynb.
    screen.blit(myfont.render(f"Ep:{episod}", True, BLACK), (5, 50))
    screen.blit(myfont.render(f"Noise:{noise_std:.3f}", True, BLACK), (5, 68))
    screen.blit(myfont.render(f"R:{ep_reward:.1f}", True, (0, 160, 0)), (5, 86))
    screen.blit(myfont.render(f"Reward:{ep_reward:.1f}", True, (0, 200, 0)), (400, 5))
    screen.blit(myfont.render(f"Speed:{car.speed:.2f}", True, BLACK), (550, 5))
    screen.blit(myfont.render(f"CP:{car.checkpoints_passed}", True, BLACK), (5, 122))
    if loss is not None:
        screen.blit(myfont.render(f"Critic loss:{loss['critic_loss']:.4f}", True, (180, 80, 0)), (5, 104))
        if loss['actor_loss'] is not None:
            screen.blit(myfont.render(f"Actor loss:{loss['actor_loss']:.4f}", True, (180, 80, 0)), (5, 140))


def run_pygame(agent, train=False, total_steps=3000, start_steps=0,
               noise_std=ACTION_NOISE, fps=60, max_episode_steps=EVAL_MAX_STEPS,
               eval_every=5000, eval_max_steps=EVAL_MAX_STEPS):
    """Наш прежний цикл отрисовки. agent передаётся готовым, веса не сбрасываются.

    train=False: просмотр без шума и изменения памяти/весов.
    train=True: продолжение обучения; начальные случайные шаги задаёт start_steps.
    Пробел — пауза, Esc/крестик — выход, S — веса, L — загрузка при просмотре.
    fps=None отключает ограничение частоты кадров.
    """
    global pygame, RENDER, screen, myfont, clock
    if pygame is None:
        import pygame as pygame_module
        pygame = pygame_module
    if total_steps <= 0 or max_episode_steps <= 0 or (fps is not None and fps <= 0):
        raise ValueError("Лимиты и fps должны быть положительными")
    previous = (RENDER, screen, myfont, clock)
    was_training = agent.actor.training
    records, updates = [], []
    try:
        pygame.init()
        RENDER = True
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("DDPG Car — обучение" if train else "DDPG Car — просмотр")
        myfont = pygame.font.SysFont('Arial', 18)
        clock = pygame.time.Clock()
        car = Car(ST_X, ST_Y)
        agent.actor.train(train)
        episod = 0
        ep_reward = 0.0
        episode_step = global_step = 0
        last_loss = None
        action = [0.0, 0.0]
        running, paused = True, False

        while running and global_step < total_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_s:
                        agent.save()
                    elif event.key == pygame.K_l:
                        if train:
                            print("Для загрузки весов завершите обучение и откройте режим просмотра.")
                        else:
                            agent.load()
                            car.reset()
                            episode_step, ep_reward = 0, 0.0
            if not running:
                break
            screen.fill(WHITE)
            for wall in walls:
                pygame.draw.line(screen, BLACK, wall[0], wall[1])

            # Флаг выключаем только на расчёте, чтобы сначала нарисовать
            # актуальный чекпоинт, затем лучи. Геометрия и физика прежние.
            RENDER = False
            car.cast_rays(walls)
            ended = False
            if not paused:
                obs = agent.get_obs(car)
                random_action = train and global_step < start_steps
                action = (np.random.uniform(-1, 1, size=N_ACTIONS).tolist() if random_action
                          else agent.choose_action(obs, noise_std=noise_std if train else 0.0))
                episode_step += 1
                next_obs, step_reward, terminated, truncated = take_step(
                    car, agent, action, episode_step, max_episode_steps
                )
                ep_reward += step_reward
                if train:
                    agent.remember(obs, action, step_reward, next_obs, terminated)
                    if global_step >= start_steps:
                        stats = agent.learn()
                        if stats is not None:
                            last_loss = stats
                            updates.append({"step": global_step + 1, **stats})
                global_step += 1
                if train:
                    agent.training_env_steps = getattr(agent, "training_env_steps", 0) + 1
                    maybe_save_best(agent, eval_every=eval_every, eval_max_steps=eval_max_steps)
                ended = terminated or truncated
            RENDER = True

            for idx, cp in enumerate(CHECKPOINTS):
                is_active = idx == car.current_checkpoint_idx
                pygame.draw.circle(screen, GREEN if is_active else (180, 180, 180), cp, 12)
                if is_active:
                    pygame.draw.circle(screen, GREEN, cp, 55, 1)
            car.cast_rays(walls)
            car.draw()
            draw_hud(episod, noise_std if train else 0.0, ep_reward, last_loss, car)
            screen.blit(myfont.render(f"Action:[{action[0]:+.2f}, {action[1]:+.2f}]", True, (0, 0, 200)), (5, 20))
            mode = "TRAIN" if train else "VIEW"
            if paused:
                mode += " / PAUSE"
            elif train and global_step <= start_steps:
                mode += " / RANDOM"
            screen.blit(myfont.render(f"{mode} Step:{global_step}  Space:pause Esc:exit", True, BLACK), (5, HEIGHT-25))
            pygame.display.flip()
            if fps is not None:
                clock.tick(fps)
            elif paused:
                clock.tick(30)

            if ended:
                records.append({"episode": episod, "step": global_step, "length": episode_step,
                                "reward": ep_reward, "checkpoints": car.checkpoints_passed,
                                "collision": terminated, "truncated": truncated, "partial": False})
                print(records[-1])
                episod += 1
                car.reset()
                episode_step, ep_reward = 0, 0.0
        if episode_step:
            records.append({"episode": episod, "step": global_step, "length": episode_step,
                            "reward": ep_reward, "checkpoints": car.checkpoints_passed,
                            "collision": False, "truncated": False, "partial": True})
        return {"episodes": records, "updates": updates,
                "evaluations": list(getattr(agent, "best_checkpoint_state", {}).get("history", []))}
    finally:
        agent.actor.train(was_training)
        pygame.quit()
        RENDER, screen, myfont, clock = previous


def diagnose_ddpg(agent, max_steps=EVAL_MAX_STEPS):
    """Один заезд без шума: сравнение Q с реальной дисконтированной наградой.
    Не меняет веса и буфер. Логи покадровых команд возвращаются в rows.
    """
    global RENDER
    old_render = RENDER
    RENDER = False
    rows = []
    try:
        car = Car(ST_X, ST_Y)
        obs = reset_car(car, agent)
        for step in range(1, max_steps + 1):
            action = agent.choose_action(obs, noise_std=0.0)
            with torch.no_grad():
                q = agent.critic(torch.FloatTensor([obs]), torch.FloatTensor([action])).item()
            next_obs, reward, terminated, truncated = take_step(car, agent, action, step, max_steps)
            rows.append({"step": step, "q": q, "reward": reward,
                         "steer": action[0], "drive": action[1], "speed": car.speed,
                         "min_ray": min(car.distance), "brake_steps": car.break_t})
            obs = next_obs
            if terminated or truncated:
                break
        discounted_return = sum((GAMMA ** i) * row['reward'] for i, row in enumerate(rows))
        print("Причина остановки:", "авария" if terminated else "внешний лимит/застревание")
        print("Шагов:", len(rows), "Чекпоинтов:", car.checkpoints_passed)
        print("Q в начале:", round(rows[0]['q'], 3), "Дисконтированная награда:", round(discounted_return, 3))
        if truncated:
            print("Заезд прерван: сумма не включает возможное продолжение; это не полный эталон Q.")
        print("Доля команд с |steer|>0.95:", sum(abs(x['steer']) > .95 for x in rows)/len(rows))
        print("Доля команд с |drive|>0.95:", sum(abs(x['drive']) > .95 for x in rows)/len(rows))
        print("Последние 5 шагов:")
        for row in rows[-5:]:
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
        return rows
    finally:
        RENDER = old_render

RUN_PYGAME = False
PYGAME_TRAIN = False  # True — продолжать обучение существующего агента с отрисовкой.
if RUN_PYGAME:
    if "training_agent" not in globals():
        raise RuntimeError("Нужен существующий training_agent: обучите его или загрузите ранее сохранённые веса.")
    pygame_log = run_pygame(
        training_agent, train=PYGAME_TRAIN, total_steps=50000,
        start_steps=0, fps=60,
        max_episode_steps=TRAIN_MAX_STEPS if PYGAME_TRAIN else EVAL_MAX_STEPS,
    )
else:
    print("Поставьте RUN_PYGAME=True. PYGAME_TRAIN=False сохраняет режим просмотра.")

RUN_DIAGNOSTICS = True
if RUN_DIAGNOSTICS:
    if "training_agent" not in globals():
        raise RuntimeError("Нужен существующий training_agent")
    diagnostic_rows = diagnose_ddpg(training_agent)

LOAD_BEST_FOR_VIEW = True
if LOAD_BEST_FOR_VIEW:
    from pathlib import Path
    state = getattr(training_agent, "best_checkpoint_state", None)
    if state is None:
        raise RuntimeError("Пока нет периодической оценки: нужно пройти 5000 шагов после включения этой функции.")
    print("Лучший шаг:", state["best_step"], "Каталог:", state["run_dir"])
    best_agent = DDPGAgent()
    best_agent.load(str(Path(state["run_dir"]) / "best_weights.pth"))
    best_metrics = evaluate_for_checkpoint(best_agent)
    print("Лучшие веса:", best_metrics)
    # Для просмотра выполните отдельно:
    # run_pygame(best_agent, train=False, total_steps=3000)
