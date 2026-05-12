using UnityEngine;

public static class GameSettings
{
    public static bool VsBot = false;         

    public static int BotPlayer = 2;          

    public static string SelectedONNXModelPath = "";

    public static bool UseGPU = false;          

    public static void Reset()
    {
        VsBot = false;
        BotPlayer = 2;
        SelectedONNXModelPath = "";
        UseGPU = false;
    }
}