using System;
using System.Collections.Generic;
using System.Linq;

namespace ConsoleGameLibrary
{

    public class ResetResult
    {
        public double[,,] Observation { get; set; }
        public Dictionary<string, object> Info { get; set; }
    }

    public class StepResult
    {
        public double[,,] Observation { get; set; }
        public float Reward { get; set; }
        public bool Terminated { get; set; }
        public bool Truncated { get; set; }
        public Dictionary<string, object> Info { get; set; }
    }

    public class ActionInfo
    {
        public int X { get; set; }
        public int Y { get; set; }
        public ShapeType Type { get; set; }
    }

    public class GameEnvironment
    {
        public Board board;
        public byte agentPlayer;

        public readonly GymnasiumObservationSpace observation_space;
        public readonly GymnasiumActionSpace action_space;

        public const int ActionSize = 162;

        public byte CurrentPlayer => board.currentPlayer;

        private int _lastAgentX = 0;
        private int _lastAgentY = 0;


        // private double lastRewardValue = 0.0;

        public Dictionary<string, object> metadata = new Dictionary<string, object>
        {
            ["render_modes"] = new[] { "human", "ansi" },
            ["name"] = "ShapeClosingGame"
        };

        public string render_mode = null;

        // self-play
        private int _opponentAction = -1;

        public GameEnvironment(byte agentPlayer = 1)
        {
            this.agentPlayer = agentPlayer;
            board = new Board();

            observation_space = new GymnasiumObservationSpace(9, 9, 18);
            action_space = new GymnasiumActionSpace(ActionSize);
        }

        // Gymnasium методы 

        public (double[,,] observation, Dictionary<string, object> info) Reset()
        {
            board = new Board();
            
            agentPlayer = (byte)(NextInt(2) + 1);

            var obs = GetObservation();
            var info = GetInfo();
            return (obs, info);
        }

        private int rngState = Environment.TickCount;

        private int NextInt(int max)
        {
            rngState = rngState * 1103515245 + 12345;
            return (rngState & 0x7fffffff) % max;
        }
        // private void OpponentMove()
        // {
        //     var legal = GetLegalActions();
        //     if (legal.Count == 0)
        //         return;

        //     int index = NextInt(legal.Count);
        //     int action = legal[index];

        //     ActionInfo move = DecodeAction(action);
        //     board.TryMakeMove(move.X, move.Y, move.Type);
        // }

        public void SetOpponentAction(int action)
        {
            _opponentAction = action;
        }

        private void OpponentMove()
        {
            var legal = GetLegalActions();
            if (legal.Count == 0)
                return;

            int action;

            if (_opponentAction != -1)
            {
                action = _opponentAction;
                _opponentAction = -1;   // используем один раз и сбрасываем
            }
            else
            {
                // случайный оппонент
                int index = NextInt(legal.Count);
                action = legal[index];
            }

            if (!legal.Contains(action))
                throw new InvalidOperationException($"Opponent returned illegal action: {action}");

            ActionInfo move = DecodeAction(action);
            board.TryMakeMove(move.X, move.Y, move.Type);
        }

        private void ComputeNewFeatures(double[,,] obs)
        {
            int playable = Board.Size - Board.Border * 2;
            int[,] wins = new int[11, 11]; // нужно для отслеживания выигрыша в один ход 
            for (int i = 0; i < playable; i++)
            {
                for (int j = 0; j < playable; j++)
                {
                    int x = i + Board.Border;
                    int y = j + Board.Border;

                    // Считаем статистику по 8 соседям
                    int squares = 0;      // количество квадратов
                    int triangles = 0;    // количество треугольников
                    int filled = 0;       // количество занятых клеток + бордер
                    int empty = 0;        // количество пустых клеток среди 8 соседей
                    int bob_x = 0;
                    int bob_y = 0;
                    bool bob_not_null = false;

                    for (int di = -1; di <= 1; di++)
                    {
                        for (int dj = -1; dj <= 1; dj++)
                        {

                            int nx = x + di;
                            int ny = y + dj;

                            if (di == 0 && dj == 0)
                            {
                                if (board.Cells[nx, ny].CurrentShape != null) bob_not_null = true;
                                continue;
                            }

                            if (nx < 0 || nx >= Board.Size || ny < 0 || ny >= Board.Size)
                            {
                                filled++;        
                                continue;
                            }

                            var shape = board.Cells[nx, ny].CurrentShape;
                            if (shape != null)
                            {
                                filled++;

                                if (shape.Type.BaseType == 2) squares++;
                                else if (shape.Type.BaseType == 1) triangles++;
                            }
                            else
                            {
                                empty++;
                                bob_x = nx;
                                bob_y = ny;
                            }
                        }
                    }

                    if (empty == 1 && bob_not_null == true)
                    {
                        if (squares == 3 || triangles == 3)
                        {
                            wins[bob_x,bob_y] = 2;
                        }
                        else if (triangles == 2 || squares == 4)
                        {
                            wins[bob_x,bob_y] = 1;
                        }
                    }

                    double filledScore = filled / 8.0;


                    obs[i, j, 9] = squares / 8.0;
                    obs[i, j, 10] = triangles / 8.0;
                    obs[i, j, 11] = filledScore;

                    // РАСЧЁТ ПОТЕНЦЕВАЛОВ

                    double potSquare = 0.0;
                    double potTriangle = 0.0;
                    double winBySquare = 0.0;
                    double winByTriangle = 0.0;

                    // Потенциал квадрата
                    int neededSquares = 4 - squares;                    
                    if (neededSquares < 0)
                    {
                        potSquare = 0; 
                    }
                    else if (neededSquares <= empty && neededSquares >= 0) 
                    {
                        potSquare = filledScore * (squares / 4.0);
                    }
                    else if (neededSquares > empty)
                    {
                        potSquare = 0;  
                    }
                    else potSquare = 0;

                    // Потенциал треугольника
                    int neededTriangles = 3 - triangles;
                    if (neededTriangles < 0)
                    {
                        potTriangle = 0; 
                    }
                    if (neededTriangles <= empty && neededTriangles >= 0)
                    {
                        potTriangle = filledScore * (triangles / 3.0);
                    }
                    else if (neededTriangles > empty)
                    {
                        potTriangle = 0;
                    }
                    else potTriangle = 0;

                    // Если клетка уже занята — обнуляем потенциал другого типа
                    var currentShape = board.Cells[x, y].CurrentShape;
                    if (currentShape != null)
                    {
                        if (currentShape.Type.BaseType == 2)      // стоит квадрат
                            potTriangle = 0.0;
                        else if (currentShape.Type.BaseType == 1) // стоит треугольник
                            potSquare = 0.0;
                    }

                    obs[i, j, 12] = Math.Min(1.0, potSquare);
                    obs[i, j, 13] = Math.Min(1.0, potTriangle);

                    // Оставшиеся ходы
                    obs[i, j, 14] = (80.0 - board.MoveCount) / 80.0;


                    // Можно ли выиграть сразу, поставив КВАДРАТ в эту клетку
                    
                    if (board.Cells[x, y].CurrentShape == null)   
                    {
                        if (filled == 8 && squares == 3)          
                            winBySquare = 1.0;
                    }

                    // if (wins[x, y] == 2) winBySquare = 1.0;
                    obs[i, j, 15] = winBySquare;   
                    

                    // Можно ли выиграть сразу, поставив ТРЕУГОЛЬНИК в эту клетку
                    if (board.Cells[x, y].CurrentShape == null)
                    {
                        if (filled == 8 && triangles == 2)        
                            winByTriangle = 1.0;
                    }
                    // if (wins[x, y] == 1) winByTriangle = 1.0;
                    obs[i, j, 16] = winByTriangle;   
                }
            }

            for (int i = 0; i < playable; i++)
            {
                for (int j = 0; j < playable; j++)
                {
                    int x = i + Board.Border;
                    int y = j + Board.Border;
                    if (wins[x,y] == 1) obs[i, j, 15] = 1.0;
                    else if (wins[x,y] == 2) obs[i, j, 16] = 1.0;
                }
            }


        }

        public void MakeOpponentMove(int action = -1)
        {
            _opponentAction = action;
            OpponentMove();
        }
        // функция передачи состояния
        public double[,,] GetObservationForPlayer(byte player)
        {
            var obs = GetObservation();  

            int playable = Board.Size - Board.Border * 2;
            for (int i = 0; i < playable; i++)
            {
                for (int j = 0; j < playable; j++)
                {
                    obs[i, j, 17] = (player == 1 ? 1.0 : 0.0);
                }
            }

            return obs;
        }

        // =============================== ЛОГИКА ХОДА АГЕНТА =============================
        public (double[,,] observation, float reward, bool terminated, bool truncated, Dictionary<string, object> info) AgentStep(int action)
        {
            if (action < 0 || action >= ActionSize)
                throw new ArgumentException("Invalid action");

            ActionInfo move = DecodeAction(action);
            bool moveSucceeded = board.TryMakeMove(move.X, move.Y, move.Type);

            if (!moveSucceeded)
            {
                return (
                    new double[9, 9, 18],
                    -10f,
                    true,
                    false,
                    GetInfo()
                );
            }

            _lastAgentX = move.X;
            _lastAgentY = move.Y;

            float Reward = (float)LocalReward(_lastAgentX, _lastAgentY);

            return (
                new double[9, 9, 18],
                Reward,
                false,                    
                false,
                GetInfo()
            );
        }

        // ================================ ЛОГИКА ХОДА СОПЕРНИКА =========================
        public (double[,,] observation, float reward, bool terminated, bool truncated, Dictionary<string, object> info) OpponentStep(int action)
        {
            // Если игра уже окончена после хода агента — не делаем ход соперника,
            // но всё равно считаем финальную награду
            if (IsGameOver())
            {
                float Reward = (float)CalculateReward(0, 0);

                return (
                    new double[9, 9, 18],
                    Reward,
                    true,
                    false,
                    GetInfo()
                );
            }

            // Игра продолжается — соперник делает нормальный ход 
            if (action < 0 || action >= ActionSize)
                throw new ArgumentException("Invalid action");

            ActionInfo move = DecodeAction(action);
            bool moveSucceeded = board.TryMakeMove(move.X, move.Y, move.Type);

            if (!moveSucceeded)
            {
                return (
                    new double[9, 9, 18],
                    -10f,
                    true,
                    false,
                    GetInfo()
                );
            }

            // Теперь, после хода соперника, считаем награду
            float finalReward = (float)CalculateReward(0, 0);
            bool terminated = IsGameOver();
            bool truncated = (board.MoveCount >= Board.MaxMoves) && !terminated;

            return (
                new double[9, 9, 18],
                finalReward,
                terminated,
                truncated,
                GetInfo()
            );
        }

        // ===================== Награда =====================

        private double LocalReward(int x, int y) {

            double reward = -0.02;
            double potential_reward = 0;
            bool idiot = false;

            for (int i = x - 1; i < x + 2; i++) {
                for (int j = y - 1; j < y + 2; j++) {

                    int a = 0;
                    int f3 = 0;
                    int f4 = 0;
                    int f_border = 0;

                    if (board.Cells[i, j].CurrentShape != null) {
                        for (int i1 = i - 1; i1 < i + 2; ++i1) {
                            for (int j1 = j - 1; j1 < j + 2; ++j1) {
                                if (board.Cells[i1, j1].CurrentShape != null && !(i1 == i && j1 == j)) {
                                    a++;
                                    switch (board.Cells[i1, j1].CurrentShape.Type.BaseType)
                                    {
                                        case 2: f4++; break;
                                        case 1: f3++; break;
                                        case 5: f_border++; break;
                                    }
                                }
                            }
                        }

                        if (a > 0) {

                            var shape = board.Cells[i, j].CurrentShape;
                            bool isOwn = shape.player == agentPlayer;

                            if (shape.Type.BaseType == 1) {
                                if (isOwn) {

                                    if (f3 < 4 && 8 - a >= 3 - f3)  potential_reward += (0.1 - 0.01*f_border) * ((f4 + f_border) / 5.0 * (f3 / 3.0));

                                    // бонус за создание сильной структуры и штраф за ее не создание.
                                    if (a == 6 && f3 == 2) reward += 0.15;
                                    if (f3 == 1 && a == 6 && board.Cells[x, y].CurrentShape.Type.BaseType != 1) reward -= 0.1;

                                    // штрафы за порчу своей области
                                    if (f3 == 2 && a == 8 && board.Cells[x, y].CurrentShape.Type.BaseType != 1) idiot = true; 
                                    if (f3 == 1 && a == 7 && board.Cells[x, y].CurrentShape.Type.BaseType != 1) reward -= 0.1;
                                    if (f3 == 0 && a == 6 && board.Cells[x, y].CurrentShape.Type.BaseType != 1) {
                                        if (f_border != 5) reward -=0.05;
                                    }

                                    if (f3 == 4 && board.Cells[x, y].CurrentShape.Type.BaseType == 1) {
                                        if (a == 8) idiot = true; // упускаем победу в 1 ход
                                        else if (a == 7) reward -= 0.12;
                                        else reward -= 0.08;
                                    }
                                }

                                else {

                                    if (f3 < 4 && 8 - a >= 3 - f3)  potential_reward -= (0.08 - 0.008*f_border) * ((f4 + f_border) / 5.0 * (f3 / 3.0));

                                    // штрафы за создание сильной структуры сопернику и бонус за ее поломку
                                    if (a == 6 && f3 == 2) reward -= 0.1;
                                    if (f3 == 1 && a == 6 && board.Cells[x, y].CurrentShape.Type.BaseType != 1) reward += 0.1;

                                    // Очень сильный штраф за закрытие области сопернику
                                    if (f3 == 3 && a == 8) {
                                        reward -= 0.8;
                                        idiot = true;
                                    }

                                    // Очень сильный штраф за создание ситуции, когда соперник побеждает в один ход
                                    if (a == 7 && (f3 == 2 || f3 == 3)) {
                                        reward -= 0.6;
                                        idiot = true;
                                    }

                                    // Бонусы за порчу области соперника. Чем позже сломали, тем лучше
                                    if (f3 == 2 && a == 8 && board.Cells[x, y].CurrentShape.Type.BaseType != 1) reward += 0.25; 
                                    if (f3 == 1 && a == 7 && board.Cells[x, y].CurrentShape.Type.BaseType != 1) reward += 0.15;
                                    if (f3 == 0 && a == 6 && board.Cells[x, y].CurrentShape.Type.BaseType != 1) reward += 0.1;

                                    if (f3 == 4 && board.Cells[x, y].CurrentShape.Type.BaseType == 1) {
                                        if (a == 8) reward += 0.25; 
                                        else if (a == 7) reward += 0.15;
                                        else reward += 0.1;
                                    }


                                }
                            }

                            if (shape.Type.BaseType == 2) {
                                if (isOwn) {
                                    if (f4 < 5 && 8 - a >= 4 - f4)  potential_reward += (0.1 - 0.01*f_border) * ((f3 + f_border) / 4.0 * (f4 / 4.0));

                                    // бонус за создание сильной структуры и штраф за ее не создание.
                                    if (a == 6 && f4 == 3) reward += 0.15;
                                    if (f4 == 2 && a == 6 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) reward -= 0.1;

                                    // штрафы за порчу своей области
                                    if (f4 == 3 && a == 8 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) idiot = true; 
                                    if (f4 == 2 && a == 7 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) reward -= 0.1;
                                    if (f4 == 1 && a == 6 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) {
                                        if (f_border != 5) reward -=0.05;
                                    }

                                    if (f4 == 5 && board.Cells[x, y].CurrentShape.Type.BaseType == 2) {
                                        if (a == 8) idiot = true; // упускаем победу в 1 ход
                                        else if (a == 7) reward -= 0.12;
                                        else reward -= 0.08;
                                    }
                                }

                                else {
                                    
                                    if (f4 < 5 && 8 - a >= 4 - f4)  potential_reward -= (0.08 - 0.008*f_border) * ((f3 + f_border) / 4.0 * (f4 / 4.0));

                                    // штрафы за создание сильной структуры сопернику и бонус за ее поломку
                                    if (a == 6 && f4 == 3) reward -= 0.1;
                                    if (f4 == 2 && a == 6 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) reward += 0.1;

                                    // Очень сильный штраф за закрытие области сопернику
                                    if (f4 == 4 && a == 8) {
                                        reward -= 0.8;
                                        idiot = true;
                                    }

                                    // Очень сильный штраф за создание ситуции, когда соперник побеждает в один ход
                                    if (a == 7 && (f4 == 3 || f4 == 4)) {
                                     reward -= 0.6;
                                     idiot = true;
                                    }

                                    // Бонусы за порчу области соперника. Чем позже сломали, тем лучше
                                    if (f4 == 3 && a == 8 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) reward += 0.25; 
                                    if (f4 == 2 && a == 7 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) reward += 0.15;
                                    if (f4 == 1 && a == 6 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) reward += 0.1;
                                    if (f4 == 0 && a == 5 && board.Cells[x, y].CurrentShape.Type.BaseType != 2) reward += 0.05;

                                    if (f4 == 5 && board.Cells[x, y].CurrentShape.Type.BaseType == 2) {
                                        if (a == 8) reward += 0.25; 
                                        else if (a == 7) reward += 0.15;
                                        else reward += 0.1;
                                    }

                                }
                            }


                        }

                    }

                }
            }

            reward = Math.Clamp(reward, -0.3, 0.3);

            potential_reward = Math.Clamp(potential_reward, -0.2, 0.2);

            reward += potential_reward;

            if (idiot == true) reward -= 0.3;

            return reward;
        }


        private double CalculateReward(int x, int y)
        {
            double reward = 0.00;

            if (IsGameOver())
            {
                Scores currentScores = GetScores();

                if (currentScores.Player1 == currentScores.Player2)
                {
                    reward += 0.0;                    // ничья
                }
                else if ((agentPlayer == 1 && currentScores.Player1 > currentScores.Player2) ||
                        (agentPlayer == 2 && currentScores.Player2 > currentScores.Player1))
                {
                    reward += 1.0;                    // победа агента
                }
                else
                {
                    // Поражение агента
                    if (board.MoveCount < 15)
                    {
                        reward += -3.5;               
                    }
                    else if (board.MoveCount < 30)
                    {
                        reward += -2.5;               
                    }
                    else
                    {
                        reward += -1.0;               // обычное поражение
                    }
                }
            }
            return reward;
        }


        // Обычные каналы
        public double[,,] GetObservation()
        {
            int playable = Board.Size - Board.Border * 2;
            double[,,] obs = new double[playable, playable, 18]; 

            for (int i = 0; i < playable; i++)
            {
                for (int j = 0; j < playable; j++)
                {
                    int absX = i + Board.Border;
                    int absY = j + Board.Border;
                    var shape = board.Cells[absX, absY].CurrentShape;

                    if (shape == null)
                    {
                        obs[i, j, 0] = 1.0;
                    }
                    else
                    {
                        int baseType = shape.Type.BaseType;
                        int player = shape.player;

                        if (player == 1)
                        {
                            obs[i, j, baseType == 2 ? 1 : 2] = 1.0;
                        }
                        else if (player == 2)
                        {
                            obs[i, j, baseType == 2 ? 3 : 4] = 1.0;
                        }
                    }

                    int own = 0, opp = 0, border = 0;

                    for (int di = -1; di <= 1; di++)
                    for (int dj = -1; dj <= 1; dj++)
                    {
                        if (di == 0 && dj == 0) continue;

                        int nx = absX + di, ny = absY + dj;

                        if (nx >= 0 && nx < Board.Size && ny >= 0 && ny < Board.Size)
                        {
                            var n = board.Cells[nx, ny].CurrentShape;
                            if (n != null)
                            {
                                if (n.player == 3) border++;
                                else if (n.player == 1) own++;
                                else opp++;
                            }
                        }
                        else border++;
                    }

                    obs[i, j, 5] = own / 8.0;
                    obs[i, j, 6] = opp / 8.0;
                    obs[i, j, 7] = border / 8.0;

                    obs[i, j, 8] = (CurrentPlayer == 1 ? 1.0 : 0.0);

                    obs[i, j, 17] = (agentPlayer == 1 ? 1.0 : 0.0);
                }
            }

            ComputeNewFeatures(obs);
            return obs;
        }

        public ActionInfo DecodeAction(int action)
        {
            int pos = action % 81;
            int x = Board.Border + (pos / 9);
            int y = Board.Border + (pos % 9);

            ShapeType type = (action < 81) ? ShapeTemplates.Cube : ShapeTemplates.TrianglePrism;

            return new ActionInfo { X = x, Y = y, Type = type };
        }

        public List<int> GetLegalActions()
        {
            var legal = new List<int>();
            for (int a = 0; a < ActionSize; a++)
            {
                ActionInfo move = DecodeAction(a);
                if (board.Cells[move.X, move.Y].CurrentShape == null)
                    legal.Add(a);
            }
            return legal;
        }

        public Scores GetScores()
        {
            return new Scores 
            { 
                Player1 = board.Player1Score, 
                Player2 = board.Player2Score 
            };
        }

        // Боб
        public bool IsValidAction(int action)
            {
                ActionInfo m = DecodeAction(action);
                return board.Cells[m.X, m.Y].CurrentShape == null;
            }

        public bool IsGameOver()
        {
            return board.MoveCount >= Board.MaxMoves || board.Player1Score != board.Player2Score;
        }

        public Dictionary<string, object> GetInfo()
        {
            Scores scores = GetScores();

            return new Dictionary<string, object>
            {
                ["player1_score"] = scores.Player1,
                ["player2_score"] = scores.Player2,

                ["move_count"] = board.MoveCount,
                ["current_player"] = CurrentPlayer,
                ["agent_player"] = agentPlayer,

                ["is_agent_turn"] = (CurrentPlayer == agentPlayer ? 1 : 0),

                // важно для masking
                // ["legal_actions"] = GetLegalActions()
            };
        }

        public bool[] action_masks()
        {
            bool[] mask = new bool[ActionSize];           
            var legal = GetLegalActions();                

            foreach (int act in legal)
            {
                mask[act] = true;
            }

            return mask;
        }

        public class GymnasiumObservationSpace
        {
            public string type => "Box";
            public float low => 0f;
            public float high => 1f;
            public int[] shape { get; }
            public string dtype => "float32";

            public GymnasiumObservationSpace(int h, int w, int c)
            {
                shape = new int[] { h, w, c };
            }
        }

        public class GymnasiumActionSpace
        {
            public string type => "Discrete";
            public int n { get; }

            public GymnasiumActionSpace(int n)
            {
                this.n = n;
            }
        }

        public GameEnvironment Clone()
        {
            var clone = new GameEnvironment(agentPlayer);
            clone.board = this.CloneBoard();
            return clone;
        }

        public Board CloneBoard()
        {
            Board clone = new Board();

            for (int i = 0; i < Board.Size; i++)
            for (int j = 0; j < Board.Size; j++)
            {
                var orig = board.Cells[i, j].CurrentShape;
                if (orig != null)
                {
                    clone.Cells[i, j].CurrentShape = new Shape
                    {
                        Type = orig.Type,
                        player = orig.player
                    };
                }
            }

            clone.Player1Score = board.Player1Score;
            clone.Player2Score = board.Player2Score;
            clone.currentPlayer = board.currentPlayer;
            clone.MoveCount = board.MoveCount;

            return clone;
        }

    }

}