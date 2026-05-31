import re
import matplotlib.pyplot as plt

TXT_PATH = "./tournament_results/tournament_28600.txt"
MAX_EPISODE = 28600

episodes = []
winrates = []

pattern = re.compile(
    r"ppo_model_(\d+)\.zip\s+\|\s+W:\d+\s+L:\d+\s+D:\d+\s+\|\s+WR:([\d\.]+)%"
)

with open(TXT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)

        if match:
            episode = int(match.group(1))
            wr = float(match.group(2))

            if episode <= MAX_EPISODE:
                episodes.append(episode)
                winrates.append(wr)

sorted_data = sorted(zip(episodes, winrates))

episodes = [x[0] for x in sorted_data]
winrates = [x[1] for x in sorted_data]

plt.figure(figsize=(14, 7))

plt.plot(
    episodes,
    winrates,
    linewidth=2
)

plt.xlabel("Model Episode")
plt.ylabel("Winrate (%)")
plt.title("Winrate vs Opponent Models")

plt.grid(True)

plt.tight_layout()
plt.show()