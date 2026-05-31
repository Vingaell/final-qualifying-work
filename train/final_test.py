import pythonnet
pythonnet.load("coreclr")

import clr
import numpy as np
import os
import gc

import gymnasium as gym

from sb3_contrib import MaskablePPO
from System.Reflection import Assembly
import System

MODELS_DIR = "./models_29000"
RESULTS_DIR = "./tournament_results"

SELECTED_MODEL_EPISODE = 28600
GAMES_PER_OPPONENT = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

dll_path = r"D:\diplom\Bob1\ConsoleGameLibrary\bin\Debug\netstandard2.1\ConsoleGameLibrary.dll"

clr.AddReference(dll_path)

assembly = Assembly.LoadFrom(dll_path)
GameEnvironmentType = assembly.GetType("ConsoleGameLibrary.GameEnvironment")

current_opponent_model = None

class GameEnvWrapper(gym.Env):

    def __init__(self):

        self.env = self._create_csharp_env()

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9, 9, 18),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(162)

        self._self_play = False

    def _create_csharp_env(self):

        ctor_arg_types = System.Array[System.Type]([System.Byte])

        constructor = GameEnvironmentType.GetConstructor(
            ctor_arg_types
        )

        args = System.Array[System.Object]([
            System.Byte(1)
        ])

        return constructor.Invoke(args)

    def set_opponent(self, model):

        global current_opponent_model

        current_opponent_model = model
        self._self_play = True

    def get_observation_for_player(self, player: int):

        obs = self.env.GetObservationForPlayer(
            System.Byte(player)
        )

        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):

        self.env.Reset()

        agent_player = int(self.env.agentPlayer)

        # EXACTLY LIKE TRAINING 
        if agent_player == 2:

            opp_player = 1

            opp_obs = self.get_observation_for_player(
                opp_player
            )

            action_masks = self.action_masks()

            action, _ = current_opponent_model.predict(
                opp_obs,
                action_masks=action_masks,
                deterministic=False
            )

            self.env.OpponentStep(int(action))

        obs = self.get_observation_for_player(
            int(self.env.agentPlayer)
        )

        info = dict(self.env.GetInfo())

        return obs, info

    def step(self, action):

        action = int(action)

        agent_player = int(self.env.agentPlayer)

        # AGENT STEP 
        agent_result = self.env.AgentStep(action)

        agent_reward = float(agent_result.Item2)

        # OPPONENT STEP 
        opp_player = 3 - agent_player

        opp_obs = self.get_observation_for_player(
            opp_player
        )

        action_masks = self.action_masks()

        opp_action, _ = current_opponent_model.predict(
            opp_obs,
            action_masks=action_masks,
            deterministic=False
        )

        result = self.env.OpponentStep(int(opp_action))

        obs = self.get_observation_for_player(agent_player)

        reward = agent_reward + float(result.Item2)

        terminated = bool(result.Item3)
        truncated = bool(result.Item4)

        info = dict(result.Item5)

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    def action_masks(self):

        masks = self.env.action_masks()

        return np.array(
            [bool(x) for x in masks],
            dtype=bool
        )

    def close(self):

        if hasattr(self.env, "Dispose"):
            self.env.Dispose()


def create_env():
    return GameEnvWrapper()

# GAME

def play_game(
    env,
    main_model,
    opponent_model,
    game_id,
    opponent_name
):

    env.set_opponent(opponent_model)

    obs, info = env.reset()

    agent_player = int(env.env.agentPlayer)

    done = False
    steps = 0

    while not done:

        action_masks = env.action_masks()

        action, _ = main_model.predict(
            obs,
            action_masks=action_masks,
            deterministic=False
        )

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        steps += 1

    p1 = info.get("player1_score", 0)
    p2 = info.get("player2_score", 0)

    if agent_player == 1:
        main_score = p1
        opp_score = p2
    else:
        main_score = p2
        opp_score = p1

    if main_score > opp_score:
        result = 1
        result_text = "WIN"

    elif main_score < opp_score:
        result = -1
        result_text = "LOSS"

    else:
        result = 0
        result_text = "DRAW"

    print(
        f"[GAME {game_id+1}] "
        f"{result_text} "
        f"| score={main_score}:{opp_score} "
        f"| env_steps={steps}"
    )

    return result, steps

# TOURNAMENT

def run_tournament():

    print("\n=== LOADING MAIN MODEL ===\n")

    main_path = os.path.join(
        MODELS_DIR,
        f"ppo_model_{SELECTED_MODEL_EPISODE}.zip"
    )

    main_model = MaskablePPO.load(main_path)

    model_files = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".zip")
        and f != f"ppo_model_{SELECTED_MODEL_EPISODE}.zip"
    ])

    print(f"FOUND {len(model_files)} MODELS\n")

    result_path = os.path.join(
        RESULTS_DIR,
        f"tournament_{SELECTED_MODEL_EPISODE}.txt"
    )

    with open(result_path, "w", encoding="utf-8") as log:

        log.write(
            f"MODEL {SELECTED_MODEL_EPISODE} vs ALL\n"
        )

        log.write(
            f"GAMES_PER_OPPONENT={GAMES_PER_OPPONENT}\n\n"
        )

        for idx, file in enumerate(model_files):

            print(
                f"\n=============================="
            )

            print(
                f"[{idx+1}/{len(model_files)}] vs {file}"
            )

            print(
                f"==============================\n"
            )

            opponent_path = os.path.join(
                MODELS_DIR,
                file
            )

            opponent = MaskablePPO.load(opponent_path)

            wins = 0
            losses = 0
            draws = 0

            total_steps = 0

            env = create_env()

            for g in range(GAMES_PER_OPPONENT):

                r, steps = play_game(
                    env,
                    main_model,
                    opponent,
                    g,
                    file
                )

                total_steps += steps

                if r == 1:
                    wins += 1

                elif r == -1:
                    losses += 1

                else:
                    draws += 1

            avg_steps = total_steps / GAMES_PER_OPPONENT

            winrate = (
                wins / GAMES_PER_OPPONENT
            ) * 100

            result_line = (
                f"{file} | "
                f"W:{wins} "
                f"L:{losses} "
                f"D:{draws} | "
                f"WR:{winrate:.2f}% | "
                f"AVG_STEPS:{avg_steps:.1f}"
            )

            print("\n" + result_line + "\n")

            log.write(result_line + "\n")
            log.flush()

            env.close()

            del opponent

            gc.collect()

        log.write("\nDONE\n")

    print("\nTOURNAMENT FINISHED")
    print(result_path)

if __name__ == "__main__":
    run_tournament()