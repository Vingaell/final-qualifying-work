using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenuUI : MonoBehaviour
{
    public void PlayLocal()
    {
        SceneManager.LoadScene("SampleScene"); 
    }

    public void PlayBot()
    {
        SceneManager.LoadScene("BotSetupScene");
    }

    public void ExitGame()
    {
        Application.Quit();
    }
}