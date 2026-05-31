import pythonnet
pythonnet.load("coreclr")


import clr
import sys
import numpy as np
import random
from System.Reflection import Assembly
import System

print("Python версия:", sys.version)

dll_path = r"D:\diplom\Bob1\ConsoleGameLibrary\bin\Debug\netstandard2.1\ConsoleGameLibrary.dll"

clr.AddReference(dll_path)
assembly = Assembly.LoadFrom(dll_path)

GameEnvironmentType = assembly.GetType("ConsoleGameLibrary.GameEnvironment")


class TestGameEnv:
    def __init__(self):
        ctor_arg_types = System.Array[System.Type]([System.Byte])
        self.constructor = GameEnvironmentType.GetConstructor(ctor_arg_types)

        args = System.Array[System.Object]([System.Byte(1)])
        self.env = self.constructor.Invoke(args)

        print("GameEnvironment создан\n")

    # BASIC CHECKS 

    def check_observation(self, obs):
        arr = np.asarray(obs, dtype=np.float32)

        print("Observation shape:", arr.shape)

        if len(arr.shape) != 3:
            print("OBS shape неверный")
            return False

        if np.isnan(arr).any():
            print("OBS содержит NaN")
            return False

        print("OBS OK")
        return True

    # FIXED SCORE HANDLING

    def check_info(self, info):
        try:
            p1 = info["player1_score"]
            p2 = info["player2_score"]

            legal = info["legal_actions"]
            turn = info.get("is_agent_turn", None)

            print(f"Scores: {p1} - {p2}")
            print(f"Legal actions: {len(legal)}")
            print(f"Agent turn: {turn}")

            return True

        except Exception as e:
            print("INFO ERROR:", repr(e))
            return False

    #  ACTION CHECK

    def check_actions(self):
        legal = [int(a) for a in self.env.GetLegalActions()]

        if len(legal) == 0:
            print("No legal actions")
            return False

        if any(a < 0 or a >= 162 for a in legal):
            print("Invalid actions detected")
            return False

        print("Actions OK:", len(legal))
        return True

    #  RESET 

    def test_reset(self):
        print("\nRESET TEST")

        reset = self.env.Reset()

        obs = np.asarray(reset.Observation, dtype=np.float32)

        self.check_observation(obs)
        self.check_info(reset.Info)

    #  STEP 

    def test_step(self):
        print("\nSTEP TEST")

        legal = [int(a) for a in self.env.GetLegalActions()]
        action = random.choice(legal)

        result = self.env.Step(action)

        print("Action:", action)
        print("Reward:", result.Reward)
        print("Terminated:", result.Terminated)
        print("Truncated:", result.Truncated)

        obs = np.asarray(result.Observation, dtype=np.float32)

        self.check_observation(obs)
        self.check_info(result.Info)

        return result

    # EPISODE 

    def run_episode(self, max_steps=300):
        print("\nEPISODE TEST")

        self.env = self.constructor.Invoke(System.Array[System.Object]([System.Byte(1)]))

        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            if self.env.IsGameOver():
                break

            legal = [int(a) for a in self.env.GetLegalActions()]
            if not legal:
                break

            action = random.choice(legal)
            result = self.env.Step(action)

            total_reward += result.Reward
            steps += 1

            info = result.Info

            p1 = info["player1_score"]
            p2 = info["player2_score"]

            print(f"{steps:03d} | A:{action:<3} | R:{result.Reward:+.3f} | {p1}-{p2}")

            if result.Terminated or result.Truncated:
                break

        print("\nRESULT")
        print("Steps:", steps)
        print("Total reward:", total_reward)
        print("Game over:", self.env.IsGameOver())

    # STRESS

    def stress_test(self, episodes=10):
        print("\nSTRESS TEST")

        for ep in range(episodes):
            print(f"\nEpisode {ep+1}")

            self.env = self.constructor.Invoke(System.Array[System.Object]([System.Byte(1)]))

            total = 0

            for _ in range(200):
                if self.env.IsGameOver():
                    break

                legal = [int(a) for a in self.env.GetLegalActions()]
                if not legal:
                    break

                action = random.choice(legal)
                result = self.env.Step(action)

                total += result.Reward

                if np.isnan(result.Reward):
                    print("NaN reward detected")
                    return

                if result.Terminated:
                    break

            print("Episode reward:", total)

    # RUN 

    def test_all(self):
        print("\nFULL TEST SUITE\n")

        self.test_reset()
        self.check_actions()
        self.test_step()
        self.run_episode()
        self.stress_test()

        print("\nDONE")


if __name__ == "__main__":
    test = TestGameEnv()
    test.test_all()