import pythonnet
pythonnet.load("coreclr")

import clr
import numpy as np
import gc
import os
import pandas as pd
import time
import random
import torch

import gymnasium as gym

from System.Reflection import Assembly
import System

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.logger import configure

# гиперпараметры

TOTAL_EPISODES = 12000
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.05

PRINT_EVERY = 25
SAVE_EVERY = 50

# ТУРНИР 
WARMUP_EPISODES = 150
GAMES_PER_PAIR = 50
MAX_POOL_SIZE = 10

dll_path = r"D:\diplom\Bob1\ConsoleGameLibrary\bin\Debug\netstandard2.1\ConsoleGameLibrary.dll"

gc.collect()
clr.AddReference(dll_path)

assembly = Assembly.LoadFrom(dll_path)
GameEnvironmentType = assembly.GetType("ConsoleGameLibrary.GameEnvironment")

current_opponent_model = None
historical_models = {}      
pool_models = []            # список моделей в пуле 
elo_ratings = {}            
model_names = {}            
problematic_models = []     # 2 проблемные модели

# GYM WRAPPER

class GameEnvWrapper(gym.Env):
    def __init__(self):                    
        self.env = self._create_csharp_env()

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(9, 9, 18),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(162)

        self._self_play = False

    def _create_csharp_env(self):
        ctor_arg_types = System.Array[System.Type]([System.Byte])
        constructor = GameEnvironmentType.GetConstructor(ctor_arg_types)
        args = System.Array[System.Object]([System.Byte(1)])
        return constructor.Invoke(args)

    def set_opponent(self, model):
        global current_opponent_model
        current_opponent_model = model
        self._self_play = True

    def set_random_opponent(self):
        global current_opponent_model
        current_opponent_model = None
        self._self_play = False

    def get_observation_for_player(self, player: int):
        obs = self.env.GetObservationForPlayer(System.Byte(player))
        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.env.Reset()

        agent_player = int(self.env.agentPlayer)

        if agent_player == 2:
            opp_player = 1
            opp_obs = self.get_observation_for_player(opp_player)

            action_masks = self.action_masks()

            if self._self_play and current_opponent_model is not None:
                action, _ = current_opponent_model.predict(
                    opp_obs, action_masks=action_masks, deterministic=False
                )
                self.env.OpponentStep(int(action))
            else:
                legal = np.where(action_masks)[0]
                if len(legal) > 0:
                    action = np.random.choice(legal)
                    self.env.OpponentStep(int(action))

        obs = self.get_observation_for_player(agent_player)
        info = dict(self.env.GetInfo())
        return obs, info

    def step(self, action):
        action = int(action)
        agent_player = int(self.env.agentPlayer)

        self.env.AgentStep(action)

        opp_player = 3 - agent_player
        opp_obs = self.get_observation_for_player(opp_player)

        action_masks = self.action_masks()

        if self._self_play and current_opponent_model is not None:
            opp_action, _ = current_opponent_model.predict(
                opp_obs, action_masks=action_masks, deterministic=False
            )
            result = self.env.OpponentStep(int(opp_action))
        else:
            legal = np.where(action_masks)[0]
            if len(legal) > 0:
                opp_action = np.random.choice(legal)
                result = self.env.OpponentStep(int(opp_action))
            else:
                result = self.env.OpponentStep(0)

        obs = self.get_observation_for_player(agent_player)
        reward = float(result.Item2)
        terminated = bool(result.Item3)
        truncated = bool(result.Item4)
        info = dict(result.Item5)

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        masks = self.env.action_masks()
        return np.array([bool(x) for x in masks], dtype=bool)

    def close(self):
        if hasattr(self.env, "Dispose"):
            self.env.Dispose()


def create_env():
    return GameEnvWrapper()

def load_random_old_model(models_dir, existing_names):
    files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]

    if not files:
        return None, None

    random.shuffle(files)

    for file in files:
        name = file.replace(".zip", "")

        # проверяем только текущий пул
        if name in existing_names:
            continue

        path = os.path.join(models_dir, file)
        loaded_model = MaskablePPO.load(path)

        return loaded_model, name

    return None, None

def normalize_name(name: str):
    return name.replace("ppo_model_", "").replace(".zip", "")

# TRAIN 

def train():
    print("Создаём окружение...")
    env = create_env()

    log_dir = "./sb3_logs"
    models_dir = "./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    new_logger = configure(log_dir, ["stdout", "csv"])

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
        device="cuda",
        tensorboard_log="./ppo_shape_tensorboard/"
    )

    model.set_logger(new_logger)

    previous_model = None
    historical_models = {}
    pool_models = []
    elo_ratings = {}
    problematic_models = []

    print("Начало обучения...\n")

    custom_log = open(os.path.join(log_dir, "sb3_custom_log.txt"), "w", encoding="utf-8")
    progress_log = open(os.path.join(log_dir, "training_progress.txt"), "w", encoding="utf-8")
    pairwise_log = open(os.path.join(log_dir, "pairwise_results.txt"), "w", encoding="utf-8")
    
    try:
        for episode in range(0, TOTAL_EPISODES):

            # ВЫБОР ОППОНЕНТА
            if episode < WARMUP_EPISODES:
                env.set_random_opponent()
            else:
                pos = episode % SAVE_EVERY
                if pos < 12:                                    # 12 self-play
                    env.set_opponent(model)
                elif pos < 26:                                  # 14 vs Top
                    if pool_models:
                        env.set_opponent(random.choice(pool_models[:6]))
                    else:
                        env.set_opponent(model)
                elif pos < 38:                                  # 14 vs problematic
                    if pool_models:
                        env.set_opponent(random.choice(problematic_models))
                    else:
                        env.set_opponent(model)
                elif pos < 46:                                  # 8 random from pool
                    if pool_models:
                        env.set_opponent(random.choice(pool_models))
                    else:
                        env.set_opponent(model)
                else:                                           # 4 fully random
                    env.set_random_opponent()

            model.learn(total_timesteps=N_STEPS, reset_num_timesteps=False)

            if episode % PRINT_EVERY == 0:
                print(f"Episode {episode} завершён")

            # СОХРАНЕНИЕ + ТУРНИР 
            if episode % SAVE_EVERY == 0 and episode > 0:
                save_path = os.path.join(models_dir, f"ppo_model_{episode}.zip")
                model.save(save_path)
                print(f"Модель сохранена → ppo_model_{episode}.zip")

                if previous_model is not None:
                    # del previous_model
                    gc.collect()

                previous_model = MaskablePPO.load(save_path)
                historical_models[episode] = previous_model
                model_names[previous_model] = f"ppo_model_{episode}"

                # ТУРНИР 
                print(f"Проведение турнира после эпизода {episode}...")
                tournament_start = time.time()

                # ТУРНИРНЫЙ ПУЛ 

                if pool_models:
                    current_pool = list(pool_models)
                else:
                    # первый турнир
                    current_pool = list(historical_models.values())

                # текущая модель обязана участвовать
                current_pool.insert(0, previous_model)

                # удаляем дубликаты по имени модели
                unique_pool = []
                seen_names = set()

                for m in current_pool:

                    name = model_names.get(m)

                    if name not in seen_names:
                        unique_pool.append(m)
                        seen_names.add(name)

                current_pool = unique_pool[:MAX_POOL_SIZE]

                # ELO обнуляется перед турниром
                elo_ratings = {m: 1500.0 for m in current_pool}

                tournament_log = []
                K = 32  # коэффициент Elo
                
                results_matrix = {}

                # ПРОВЕДЕНИЕ ТУРНИРА
                for round_num in range(GAMES_PER_PAIR):
                    for i in range(len(current_pool)):
                        for j in range(i + 1, len(current_pool)):
                            m1 = current_pool[i]
                            m2 = current_pool[j]

                            env_game = create_env()

                            # СМЕНА РОЛЕЙ 
                            if round_num < GAMES_PER_PAIR // 2:
                                agent_model = m1
                                opponent_model = m2
                            else:
                                agent_model = m2
                                opponent_model = m1

                            env_game.set_opponent(opponent_model)

                            obs, info = env_game.reset()
                            done = False

                            while not done:
                                action_masks = env_game.action_masks()

                                action, _ = agent_model.predict(
                                    obs,
                                    action_masks=action_masks,
                                    deterministic=True
                                )

                                obs, reward, terminated, truncated, info = env_game.step(action)
                                done = terminated or truncated

                            agent_player = int(env_game.env.agentPlayer)

                            p1 = info.get("player1_score", 0)
                            p2 = info.get("player2_score", 0)

                            if agent_player == 1:
                                agent_score = p1
                                opponent_score = p2
                            else:
                                agent_score = p2
                                opponent_score = p1

                            if agent_score > opponent_score:
                                agent_result = 1.0
                            elif opponent_score > agent_score:
                                agent_result = 0.0
                            else:
                                agent_result = 0.5

                            if agent_model == m1:
                                score1 = agent_result
                            else:
                                score1 = 1.0 - agent_result if agent_result != 0.5 else 0.5

                            # Обновление ELO
                            elo1 = elo_ratings[m1]
                            elo2 = elo_ratings[m2]
                            expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400.0))

                            elo_ratings[m1] = elo1 + K * (score1 - expected1)
                            elo_ratings[m2] = elo2 + K * ((1 - score1) - (1 - expected1))

                            # Логируем результат каждой игры
                            name1 = model_names[m1]
                            name2 = model_names[m2]
                            result_line = f"Round {round_num+1} | {name1} vs {name2} | {'Win' if score1 == 1 else 'Loss' if score1 == 0 else 'Draw'}"
                            tournament_log.append(result_line)
                            
                            # RESULTS MATRIX 
                            pair_key = (name1, name2)

                            if pair_key not in results_matrix:
                                results_matrix[pair_key] = [0, 0, 0]  # wins1, wins2, draws

                            if score1 == 1:
                                results_matrix[pair_key][0] += 1
                            elif score1 == 0:
                                results_matrix[pair_key][1] += 1
                            else:
                                results_matrix[pair_key][2] += 1
                                
                            env_game.close()
                            del env_game

                tournament_time = time.time() - tournament_start

                # ФОРМИРОВАНИЕ ПУЛА 
                # 6 лучших по ELO
                sorted_by_elo = sorted(current_pool, key=lambda m: elo_ratings[m], reverse=True)
                new_pool = list(sorted_by_elo[:6])
                
                # обучать победителя турнира или нет??? абаюнда
                winner = sorted_by_elo[0]
                
                # жесткое обновление весов
                # model.policy.load_state_dict(winner.policy.state_dict())
                
                
                # мягкое обновление весов
                tau = 0.4

                for param, w_param in zip(model.policy.parameters(), winner.policy.parameters()):
                    param.data.copy_((1 - tau) * param.data + tau * w_param.data)

                # Проблемные модели для лидера
                leader = sorted_by_elo[0]
                leader_name = model_names[leader]

                leader_winrates = {}

                for (n1, n2), (w1, w2, d) in results_matrix.items():

                    if n1 == leader_name:
                        opponent_name = n2
                        winrate = w1 / GAMES_PER_PAIR

                    elif n2 == leader_name:
                        opponent_name = n1
                        winrate = w2 / GAMES_PER_PAIR

                    else:
                        continue

                    leader_winrates[opponent_name] = winrate
                # 2 самые проблемные
                problematic = sorted(leader_winrates.items(), key=lambda x: x[1])[:2]
                problematic = [p[0] for p in problematic]
                problematic_models = []

                for p_name in problematic:
                    for m in current_pool:

                        if model_names[m] == p_name:

                            # добавляем в problematic для обучения
                            problematic_models.append(m)

                            # добавляем в турнирный пул только если ещё нет
                            if m not in new_pool:
                                new_pool.append(m)

                # Дополняем случайными старыми до 10
                existing_names = set(model_names[m] for m in new_pool)

                attempts = 0
                max_attempts = 50

                while len(new_pool) < MAX_POOL_SIZE and attempts < max_attempts:

                    random_old, random_name = load_random_old_model(
                        models_dir,
                        existing_names
                    )

                    if random_old is None:
                        break

                    if random_name not in existing_names:
                        model_names[random_old] = random_name
                        new_pool.append(random_old)
                        existing_names.add(random_name)
                    else:
                        del random_old

                    attempts += 1

                pool_models = new_pool
                
                # ОЧИСТКА НЕИСПОЛЬЗУЕМЫХ МОДЕЛЕЙ 

                models_to_keep = set(pool_models)

                for ep in list(historical_models.keys()):
                    model_ref = historical_models[ep]

                    if model_ref not in models_to_keep:
                        del historical_models[ep]
                        # del model_ref

                gc.collect()
                torch.cuda.empty_cache()
                
                
                # ЛОГ РЕЗУЛЬТАТОВ ПАР 

                pairwise_log.write(f"\n=== ТУРНИР после эпизода {episode} ===\n")

                for (n1, n2), (w1, w2, d) in sorted(results_matrix.items()):

                    total_games = w1 + w2 + d

                    wr1 = (w1 / total_games) * 100 if total_games > 0 else 0
                    wr2 = (w2 / total_games) * 100 if total_games > 0 else 0

                    pairwise_log.write(
                        f"{n1} vs {n2} | "
                        f"{w1}-{w2}-{d} | "
                        f"WR: {wr1:.1f}% / {wr2:.1f}%\n"
                    )

                pairwise_log.write("=" * 90 + "\n\n")
                pairwise_log.flush()

                # ЗАПИСЬ В ЛОГ 
                custom_log.write(f"\n=== ТУРНИР после эпизода {episode} ===\n")
                custom_log.write(f"Время турнира: {tournament_time:.1f} секунд\n")

                # custom_log.write("Результаты матчей:\n")
                # for line in tournament_log:
                #     custom_log.write(line + "\n")

                custom_log.write("\nELO после турнира:\n")
                for m in sorted_by_elo:
                    name = model_names[m]
                    custom_log.write(f"{name}: {elo_ratings[m]:.0f}\n")

                custom_log.write("="*90 + "\n\n")
                custom_log.flush()

                print(f"Турнир завершён за {tournament_time:.1f} сек.")

            # Запись progress
            csv_path = os.path.join(log_dir, "progress.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        last_row = df.iloc[-1]
                        step = len(df)
                        timesteps = int(last_row.get('total_timesteps', 0))
                        ep_rew_mean = float(last_row.get('rollout/ep_rew_mean', 0))
                        ep_len_mean = float(last_row.get('rollout/ep_len_mean', 0))
                        time_elapsed = float(last_row.get('time/time_elapsed', 0))

                        progress_line = f"[STEP {step}] | timesteps={timesteps} | ep_rew_mean={ep_rew_mean:.3f} | ep_len_mean={ep_len_mean:.3f} | time_elapsed={time_elapsed:.0f}\n"
                        progress_log.write(progress_line)
                        progress_log.flush()
                except:
                    pass

    finally:
        custom_log.close()
        progress_log.close()
        pairwise_log.close()

    model.save(os.path.join(models_dir, "ppo_model_final.zip"))
    print("\nОбучение завершено!")


if __name__ == "__main__":
    train()