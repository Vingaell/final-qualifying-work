import os
import torch
import numpy as np
import onnx

from sb3_contrib import MaskablePPO

MODELS_DIR = "./models"
ONNX_DIR = "./models_onnx"

os.makedirs(ONNX_DIR, exist_ok=True)


class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()

    def forward(self, obs):
        with torch.no_grad():
            obs = obs.float()

            # 1. Features extractor
            features = self.policy.features_extractor(obs)

            # 2. MlpExtractor
            if hasattr(self.policy, 'mlp_extractor') and self.policy.mlp_extractor is not None:
                latent_pi, _ = self.policy.mlp_extractor(features)
            else:
                latent_pi = features

            # 3. Action net → logits
            logits = self.policy.action_net(latent_pi)

            return logits


def export_model(model_path, onnx_path):
    print(f"\n{'='*80}")
    print(f"Экспорт: {os.path.basename(model_path)}")
    print(f"{'='*80}")

    try:
        model = MaskablePPO.load(model_path, device="cpu")
        policy = model.policy

        wrapper = PolicyWrapper(policy)

        dummy_input = torch.zeros((1, 9, 9, 18), dtype=torch.float32)

        print("Запуск экспорта...")

        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            input_names=["obs"],
            output_names=["logits"],
            opset_version=13,
            dynamic_axes={
                "obs": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            do_constant_folding=True,
        )

        print(f"ONNX сохранён → {onnx_path}")

        # Проверка
        onnx_model = onnx.load(onnx_path)
        output_shape = onnx_model.graph.output[0].type.tensor_type.shape.dim
        dims = [dim.dim_value if dim.HasField('dim_value') else 'dynamic' for dim in output_shape]
        
        print(f"Logits shape: {dims}")

        if len(dims) > 0 and dims[-1] == 162:
            print(" Модель экспортирована корректно (162 действий).")
        else:
            print(f"Размер выхода неправильный: {dims}")

    except Exception as e:
        print(f"Ошибка экспорта: {e}")


def main():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]
    print(f"Найдено моделей: {len(files)}\n")

    for file in files:
        model_path = os.path.join(MODELS_DIR, file)
        onnx_path = os.path.join(ONNX_DIR, file.replace(".zip", ".onnx"))
        export_model(model_path, onnx_path)

    print("\nЭкспорт завершён!")


if __name__ == "__main__":
    main()