using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

using ConsoleGameLibrary;

public class OnnxModelRunner : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;

    private readonly GameEnvironment _gameEnv;

    public OnnxModelRunner(string modelPath, GameEnvironment gameEnv)
    {
        try
        {
            var sessionOptions = new SessionOptions();
            _session = new InferenceSession(modelPath, sessionOptions);

            _inputName = _session.InputMetadata.Keys.FirstOrDefault();
            _outputName = _session.OutputMetadata.Keys.FirstOrDefault();

            _gameEnv = gameEnv;

            Debug.Log($"[ONNX] МОДЕЛЬ ЗАГРУЖЕНА УСПЕШНО");
            Debug.Log($"[ONNX] Input name : {_inputName}");
            Debug.Log($"[ONNX] Output name: {_outputName}");

            if (_session.OutputMetadata.TryGetValue(_outputName, out var meta))
            {
                var dims = meta.Dimensions.ToArray();
                Debug.Log($"[ONNX] Output shape: [{string.Join(", ", dims)}]");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[ONNX] Ошибка загрузки модели: {e.Message}");
            throw;
        }
    }

    public ActionInfo Predict(byte botPlayer, bool[] mask)
    {
        if (mask == null || mask.Length != 162)
        {
            Debug.LogError("[ONNX] Неверная маска действий");
            return null;
        }

        int legalCount = mask.Count(m => m);
        Debug.Log($"[ONNX Predict] Legal actions: {legalCount}/162");

        // OBS 
        double[,,] doubleObs = _gameEnv.GetObservationForPlayer(botPlayer);

        var inputTensor = new DenseTensor<float>(new[] { 1, 9, 9, 18 });

        for (int x = 0; x < 9; x++)
        for (int y = 0; y < 9; y++)
        for (int c = 0; c < 18; c++)
            inputTensor[0, x, y, c] = (float)doubleObs[x, y, c];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        try
        {
            using var results = _session.Run(inputs);

            if (results == null || results.Count == 0)
            {
                Debug.LogError("[ONNX] Run вернул пустой результат!");
                return null;
            }

            var outputNamedValue = results.FirstOrDefault(r => r.Name == _outputName);

            if (outputNamedValue == null)
            {
                Debug.LogError($"[ONNX] Output '{_outputName}' НЕ НАЙДЕН!");
                Debug.Log($"[ONNX] Доступные outputs: {string.Join(", ", results.Select(r => r.Name))}");
                return null;
            }

            Debug.Log($"[ONNX] Найден output '{outputNamedValue.Name}'");

            float[] logits = null;

            try
            {
                var tensor = outputNamedValue.AsTensor<float>();
                logits = tensor.ToArray();

                Debug.Log("[ONNX] Output прочитан как float");
            }
            catch (Exception)
            {
                Debug.LogWarning("[ONNX] float не подошёл → fallback через object");

                var raw = outputNamedValue.Value;

                Debug.Log($"[ONNX] Реальный тип: {raw.GetType()}");

                var enumerable = raw as System.Collections.IEnumerable;
                if (enumerable == null)
                {
                    Debug.LogError("[ONNX] Output не IEnumerable");
                    return null;
                }

                var list = new List<float>();

                foreach (var v in enumerable)
                {
                    list.Add(Convert.ToSingle(v));
                }

                logits = list.ToArray();

                Debug.Log($"[ONNX] Fallback logits: {logits.Length}");
            }

            // ПРОВЕРКА 
            if (logits == null || logits.Length == 0)
            {
                Debug.LogError("[ONNX] logits пуст!");
                return null;
            }

            Debug.Log($"[ONNX Predict] Model returned {logits.Length} logits");

            // MASK
            for (int i = 0; i < logits.Length; i++)
            {
                if (i < mask.Length && !mask[i])
                    logits[i] = -10000f;
            }

            // ===================== SOFTMAX =====================
            float maxLogit = logits.Max();
            float sumExp = 0f;

            float[] probs = new float[logits.Length];

            for (int i = 0; i < logits.Length; i++)
            {
                probs[i] = Mathf.Exp(logits[i] - maxLogit);
                sumExp += probs[i];
            }

            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sumExp;
            }

            // ===================== FILTER + LOG =====================
            List<int> candidates = new List<int>();

            Debug.Log("[ONNX Predict] Candidate actions:");

            for (int i = 0; i < probs.Length; i++)
            {
                if (i < mask.Length && mask[i] && probs[i] > 0.03f)
                {
                    candidates.Add(i);
                    Debug.Log($"Action {i} | prob = {probs[i]:F4}");
                }
            }

            // fallback если всё отфильтровалось
            if (candidates.Count == 0)
            {
                int maxIdx = 0;
                float maxP = -1f;

                for (int i = 0; i < probs.Length; i++)
                {
                    if (i < mask.Length && mask[i] && probs[i] > maxP)
                    {
                        maxP = probs[i];
                        maxIdx = i;
                    }
                }

                candidates.Add(maxIdx);
            }

            // ===================== SAMPLING =====================
            int chosen = candidates[UnityEngine.Random.Range(0, candidates.Count)];

            Debug.Log($"[ONNX Predict] Bot chose action #{chosen} (prob={probs[chosen]:F4})");

            return _gameEnv.DecodeAction(chosen);

        }
        catch (Exception e)
        {
            Debug.LogError($"[ONNX] Ошибка инференса: {e.Message}\nStack: {e.StackTrace}");
            return null;
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}