using UnityEngine;
using TMPro;
using ConsoleGameLibrary;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;

public class BoardUI : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject cellPrefab;
    public GameObject[] piecePrefabs;        // 0 - Cube, 1 - TrianglePrism

    [Header("Материал клетки")]
    public Material cellMaterial;

    [Header("UI")]
    public TextMeshProUGUI player1Text;
    public TextMeshProUGUI player2Text;
    public TextMeshProUGUI gameResultText;     // текст результата игры
    public GameObject exitButton;              // Кнопка главного меню

    private GameObject[,] cells;
    private Board gameBoard;
    private GameObject currentPiecePrefab;

    private OnnxModelRunner _botRunner;
    private GameEnvironment _gameEnv;
    private bool _isVsBot;
    private byte _humanPlayer;
    private byte _botPlayer;

    private bool _gameOver = false;

    void Start()
    {
        _isVsBot = GameSettings.VsBot;

        if (_isVsBot)
        {
            _botPlayer = (byte)GameSettings.BotPlayer;
            _humanPlayer = (byte)(3 - _botPlayer);

            _gameEnv = new GameEnvironment(_botPlayer);
            _botRunner = new OnnxModelRunner(GameSettings.SelectedONNXModelPath, _gameEnv);

            gameBoard = _gameEnv.board;

            Debug.Log($"[BoardUI] Режим против бота. Ты играешь за Player {_humanPlayer}, бот — за Player {_botPlayer}");
        }
        else
        {
            gameBoard = new Board();
        }

        int size = Board.Size;
        int border = Board.Border;

        cells = new GameObject[size, size];

        for (int x = border; x < size - border; x++)
        {
            for (int z = border; z < size - border; z++)
            {
                GameObject cell = Instantiate(cellPrefab, new Vector3(x, 0, z), Quaternion.identity);
                cells[x, z] = cell;

                if (cell.GetComponent<Collider>() == null)
                    cell.AddComponent<BoxCollider>();

                cell.GetComponent<Renderer>().material = cellMaterial;
            }
        }

        if (piecePrefabs.Length > 0)
            currentPiecePrefab = piecePrefabs[0];

        // Настройка UI результата
        if (gameResultText != null)
            gameResultText.gameObject.SetActive(false);

        DrawGridLines();
        UpdateScoreUI();
        PrintScoreToConsole();

        if (_isVsBot && gameBoard.currentPlayer == _botPlayer)
        {
            MakeBotMove();
        }
    }

    void Update()
    {
        if (Mouse.current == null || _gameOver) return;

        if (_isVsBot && gameBoard.currentPlayer != _humanPlayer)
            return;

        if (Mouse.current.leftButton.wasPressedThisFrame)
        {
            Ray ray = Camera.main.ScreenPointToRay(Mouse.current.position.ReadValue());

            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                for (int x = Board.Border; x < Board.Size - Board.Border; x++)
                {
                    for (int z = Board.Border; z < Board.Size - Board.Border; z++)
                    {
                        if (cells[x, z] != null && hit.collider.gameObject == cells[x, z])
                        {
                            ClickCell(x, z);
                            return;
                        }
                    }
                }
            }
        }
    }

    void ClickCell(int x, int z)
    {
        if (_gameOver || currentPiecePrefab == null) return;

        byte playerBeforeMove = gameBoard.currentPlayer;

        bool success = gameBoard.TryMakeMove(x, z, GetShapeFromPrefab(currentPiecePrefab));

        if (!success)
        {
            Debug.LogWarning($"Invalid move at ({x}, {z})");
            return;
        }

        GameObject piece = Instantiate(currentPiecePrefab, new Vector3(x, 0.5f, z), Quaternion.identity);
        
        Renderer r = piece.GetComponent<Renderer>();
        if (r != null)
        {
            r.material.color = (playerBeforeMove == 1) ? Color.white : Color.black;
        }

        UpdateScoreUI();
        PrintScoreToConsole();

        CheckGameOver();

        if (!_gameOver && _isVsBot && gameBoard.currentPlayer == _botPlayer)
        {
            MakeBotMove();
        }
    }

    private void MakeBotMove()
    {
        if (!_isVsBot || _botRunner == null || _gameOver) return;

        _gameEnv.board = gameBoard;

        bool[] mask = _gameEnv.action_masks();

        ActionInfo aiMove = _botRunner.Predict(_botPlayer, mask);

        if (aiMove == null)
        {
            Debug.LogWarning("Bot could not find a valid move!");
            return;
        }

        byte playerBefore = gameBoard.currentPlayer;

        bool success = gameBoard.TryMakeMove(aiMove.X, aiMove.Y, aiMove.Type);
        
        if (!success)
        {
            Debug.LogWarning($"Bot tried invalid move at ({aiMove.X}, {aiMove.Y})");
            return;
        }

        GameObject prefabToUse = (aiMove.Type == ShapeTemplates.Cube) 
            ? piecePrefabs[0] 
            : (piecePrefabs.Length > 1 ? piecePrefabs[1] : piecePrefabs[0]);

        GameObject piece = Instantiate(prefabToUse, new Vector3(aiMove.X, 0.5f, aiMove.Y), Quaternion.identity);
        
        Renderer r = piece.GetComponent<Renderer>();
        if (r != null)
        {
            r.material.color = (playerBefore == 1) ? Color.white : Color.black;
        }

        UpdateScoreUI();
        PrintScoreToConsole();

        CheckGameOver();
    }

    private void CheckGameOver()
    {
        if (_gameEnv == null)
        {
            // Для PvP режима
            if (gameBoard.MoveCount >= Board.MaxMoves || gameBoard.Player1Score != gameBoard.Player2Score)
            {
                ShowGameResult();
            }
        }
        else
        {
            if (_gameEnv.IsGameOver())
            {
                ShowGameResult();
            }
        }
    }

    private void ShowGameResult()
    {
        _gameOver = true;

        if (gameResultText == null) return;

        gameResultText.gameObject.SetActive(true);

        if (gameBoard.Player1Score > gameBoard.Player2Score)
            gameResultText.text = "Победа Белых (Player 1)!";
        else if (gameBoard.Player2Score > gameBoard.Player1Score)
            gameResultText.text = "Победа Чёрных (Player 2)!";
        else
            gameResultText.text = "Ничья!";

        gameResultText.color = Color.yellow;
    }

    // Вспомогательные методы 

    ShapeType GetShapeFromPrefab(GameObject prefab)
    {
        if (prefab == null) 
            return ShapeTemplates.Cube;

        string name = prefab.name.ToLower();

        if (name.Contains("cube") || name.Contains("square"))
            return ShapeTemplates.Cube;

        if (name.Contains("triangle"))
            return ShapeTemplates.TrianglePrism;

        Debug.LogWarning("Unknown piece prefab: " + prefab.name);
        return ShapeTemplates.Cube;
    }

    public void SetCurrentPiece(GameObject prefab)
    {
        currentPiecePrefab = prefab;
    }

    void UpdateScoreUI()
    {
        if (player1Text != null)
            player1Text.text = "Player 1: " + gameBoard.Player1Score;

        if (player2Text != null)
            player2Text.text = "Player 2: " + gameBoard.Player2Score;
    }

    void PrintScoreToConsole()
    {
        Debug.Log($"=== Current Scores ===\nPlayer 1: {gameBoard.Player1Score}\nPlayer 2: {gameBoard.Player2Score}\nMove Count: {gameBoard.MoveCount}");
    }

    // Кнопка выхода в меню
    public void ExitToMainMenu()
    {
        SceneManager.LoadScene("MainMenu");
    }

    void DrawGridLines()
    {
        int size = Board.Size;
        int border = Board.Border;
        Color lineColor = Color.black;
        float lineWidth = 0.05f;

        for (int x = border; x < size - border; x++)
        {
            for (int z = border; z < size - border; z++)
            {
                Vector3 cellPos = new Vector3(x, 0.01f, z);

                if (x < size - border - 1)
                {
                    GameObject lineObj = new GameObject($"Line_Right_{x}_{z}");
                    LineRenderer lr = lineObj.AddComponent<LineRenderer>();
                    lr.positionCount = 2;
                    lr.SetPosition(0, cellPos + new Vector3(0.5f, 0.1f, -0.5f));
                    lr.SetPosition(1, cellPos + new Vector3(0.5f, 0.1f, 0.5f));
                    lr.startWidth = lr.endWidth = lineWidth;
                    lr.material = new Material(Shader.Find("Unlit/Color"));
                    lr.material.color = lineColor;
                }

                if (z < size - border - 1)
                {
                    GameObject lineObj = new GameObject($"Line_Forward_{x}_{z}");
                    LineRenderer lr = lineObj.AddComponent<LineRenderer>();
                    lr.positionCount = 2;
                    lr.SetPosition(0, cellPos + new Vector3(-0.5f, 0.1f, 0.5f));
                    lr.SetPosition(1, cellPos + new Vector3(0.5f, 0.1f, 0.5f));
                    lr.startWidth = lr.endWidth = lineWidth;
                    lr.material = new Material(Shader.Find("Unlit/Color"));
                    lr.material.color = lineColor;
                }
            }
        }
    }

    void OnDestroy()
    {
        _botRunner?.Dispose();
    }
}