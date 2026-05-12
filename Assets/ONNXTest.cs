using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using System;

public class ONNXTest : MonoBehaviour
{
    void Start()
    {
        try
        {
            Debug.Log("ONNX Runtime успешно подключён!");

            string modelPath = "Assets/Models/ppo_model_2200.onnx";   

            using (var session = new InferenceSession(modelPath))
            {
                Debug.Log("Модель успешно загружена!");
                Debug.Log($"Количество входов: {session.InputMetadata.Count}");
                Debug.Log($"Количество выходов: {session.OutputMetadata.Count}");

                foreach (var input in session.InputMetadata)
                {
                    Debug.Log($"Вход: {input.Key} | Форма: {string.Join(",", input.Value.Dimensions)}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Ошибка при работе с ONNX Runtime:");
            Debug.LogError(e.Message);
            if (e.InnerException != null)
                Debug.LogError("Inner: " + e.InnerException.Message);
        }
    }
}