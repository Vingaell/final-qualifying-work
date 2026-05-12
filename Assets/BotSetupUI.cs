using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;
using System.IO;
using System.Collections.Generic;

public class BotSetupUI : MonoBehaviour
{
    public TMP_Dropdown modelDropdown;
    public TMP_Dropdown playerDropdown;

    private List<string> modelPaths = new List<string>();

    private void Start()
    {
        LoadAvailableONNXModels();
    }

    private void LoadAvailableONNXModels()
    {
        modelDropdown.ClearOptions();
        modelPaths.Clear();

        string modelsFolder = Path.Combine(Application.dataPath, "Models");

        if (!Directory.Exists(modelsFolder))
        {
            Debug.LogError("Папка Assets/Models не найдена!");
            return;
        }

        string[] files = Directory.GetFiles(modelsFolder, "*.onnx");

        List<string> options = new List<string>();

        foreach (string file in files)
        {
            options.Add(Path.GetFileName(file));
            modelPaths.Add(file);
        }

        if (options.Count == 0)
            options.Add("Нет моделей");

        modelDropdown.AddOptions(options);

        Debug.Log("Загружено моделей: " + options.Count);
    }

    public void StartGame()
    {
        if (modelPaths.Count == 0 || modelDropdown.value >= modelPaths.Count)
        {
            Debug.LogError("Модель не выбрана!");
            return;
        }

        GameSettings.VsBot = true;
        GameSettings.BotPlayer = (playerDropdown.value == 0) ? 1 : 2;
        GameSettings.SelectedONNXModelPath = modelPaths[modelDropdown.value];

        Debug.Log("Запуск игры против бота. Модель: " + GameSettings.SelectedONNXModelPath);

        SceneManager.LoadScene("SampleScene");
    }

    public void Back()
    {
        SceneManager.LoadScene("MainMenu");
    }
}