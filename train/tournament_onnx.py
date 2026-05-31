import pythonnet
pythonnet.load("coreclr")

import clr
import numpy as np
import os
import time
import random

from System.Reflection import Assembly
import System

import onnxruntime as ort


ONNX_FOLDER = "./fight"   
GAMES_PER_PAIR = 500
OUTPUT_FILE = "tournament_onnx_results.txt"


print(f"Поиск ONNX моделей в папке: {ONNX_FOLDER}\n")

# Загружаем DLL
dll_path = r"D:\diplom\Bob1\ConsoleGameLibrary\bin\Debug\netstandard2.1\ConsoleGameLibrary.dll"
clr.AddReference(dll_path)
assembly = Assembly.LoadFrom(dll_path)
GameEnvironmentType = assembly.GetType("ConsoleGameLibrary.GameEnvironment")


# GameEnvWrapper 
class GameEnvWrapper:
    def __init__(self):
        self.env = self._create_csharp_env()
        self.current_opponent = None
        self._self_play = False

    def _create_csharp_env(self):
        ctor_arg_types = System.Array[System.Type]([System.Byte])
        constructor = GameEnvironmentType.GetConstructor(ctor_arg_types)
        args = System.Array[System.Object]([System.Byte(1)])
        return constructor.Invoke(args)

    def set_opponent(self, onnx_session):
        self.current_opponent = onnx_session
        self._self_play = True

    def reset(self):
        self.env.Reset()
        agent_player = int(self.env.agentPlayer)

        if agent_player == 2:
            opp_player = 1
            opp_obs = self.get_observation_for_player(opp_player)
            action_masks = self.action_masks()

            if self._self_play and self.current_opponent is not None:
                input_name = self.current_opponent.get_inputs()[0].name
                logits = self.current_opponent.run(None, {input_name: opp_obs.astype(np.float32)[np.newaxis, ...]})[0][0]
                logits = np.where(action_masks, logits, -1e9)
                action = int(np.argmax(logits))
                self.env.OpponentStep(action)
            else:
                legal = np.where(action_masks)[0]
                action = np.random.choice(legal) if len(legal) > 0 else 0
                self.env.OpponentStep(action)

        obs = self.get_observation_for_player(agent_player)
        info = dict(self.env.GetInfo())
        return obs, info

    def step(self, action):
        self.env.AgentStep(action)

        opp_player = 3 - int(self.env.agentPlayer)
        opp_obs = self.get_observation_for_player(opp_player)
        action_masks = self.action_masks()

        if self._self_play and self.current_opponent is not None:
            input_name = self.current_opponent.get_inputs()[0].name
            logits = self.current_opponent.run(None, {input_name: opp_obs.astype(np.float32)[np.newaxis, ...]})[0][0]
            logits = np.where(action_masks, logits, -1e9)
            opp_action = int(np.argmax(logits))
            result = self.env.OpponentStep(opp_action)
        else:
            legal = np.where(action_masks)[0]
            opp_action = np.random.choice(legal) if len(legal) > 0 else 0
            result = self.env.OpponentStep(opp_action)

        obs = self.get_observation_for_player(int(self.env.agentPlayer))
        reward = float(result.Item2)
        terminated = bool(result.Item3)
        truncated = bool(result.Item4)
        info = dict(result.Item5)

        return obs, reward, terminated, truncated, info

    def get_observation_for_player(self, player: int):
        obs = self.env.GetObservationForPlayer(System.Byte(player))
        return np.asarray(obs, dtype=np.float32)

    def action_masks(self):
        masks = self.env.action_masks()
        return np.array([bool(x) for x in masks], dtype=bool)


# ТУРНИР 
print("Загрузка ONNX моделей...\n")

onnx_files = [f for f in os.listdir(ONNX_FOLDER) if f.endswith(".onnx")]
onnx_files.sort()

if len(onnx_files) < 2:
    print("Ошибка: В папке fight_onnx должно быть минимум 2 модели!")
    exit()

sessions = {}
for file in onnx_files:
    path = os.path.join(ONNX_FOLDER, file)
    print(f"Загрузка: {file}")
    sessions[file] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])

print(f"\nЗапуск турнира: {len(onnx_files)} моделей, по {GAMES_PER_PAIR} игр на пару\n")

results = []
total_pairs = len(onnx_files) * (len(onnx_files) - 1) // 2
pair_count = 0

for i, m1_name in enumerate(onnx_files):
    for j, m2_name in enumerate(onnx_files):
        if i >= j:
            continue

        pair_count += 1
        sess1 = sessions[m1_name]
        sess2 = sessions[m2_name]

        wins1 = wins2 = draws = 0
        print(f"[{pair_count}/{total_pairs}] {m1_name} vs {m2_name} ... ", end="")

        for game in range(GAMES_PER_PAIR):
            env = GameEnvWrapper()
            env.set_opponent(sess2)   # m1 — агент, m2 — оппонент

            obs, info = env.reset()
            done = False

            while not done:
                input_name = sess1.get_inputs()[0].name
                logits = sess1.run(None, {input_name: obs.astype(np.float32)[np.newaxis, ...]})[0][0]

                action_masks = env.action_masks()
                logits = np.where(action_masks, logits, -1e9)
                action = int(np.argmax(logits))

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            p1 = info.get("player1_score", 0)
            p2 = info.get("player2_score", 0)

            if p1 > p2:
                wins1 += 1
            elif p2 > p1:
                wins2 += 1
            else:
                draws += 1

        winrate1 = wins1 / GAMES_PER_PAIR * 100
        winrate2 = wins2 / GAMES_PER_PAIR * 100

        line = f"{m1_name} vs {m2_name} | {wins1:3d}-{wins2:3d} ({draws:2d} draws) | WR: {winrate1:5.1f}% - {winrate2:5.1f}%"
        results.append(line)
        print("Готово")

# СОХРАНЕНИЕ 
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("=== ONNX ТУРНИРНАЯ ТАБЛИЦА ===\n\n")
    for line in results:
        f.write(line + "\n")

print("\n" + "="*90)
print(f"ТУРНИР ЗАВЕРШЁН! Результаты сохранены в: {OUTPUT_FILE}")
print("="*90)

for line in results:
    print(line)