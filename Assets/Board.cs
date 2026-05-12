using System;
using System.Collections.Generic;

namespace ConsoleGameLibrary
{
    public class Cell
    {
        public Shape CurrentShape;
    }

    public class Scores
    {
        public int Player1 { get; set; }
        public int Player2 { get; set; }
    }

    public class Board
    {
        public const int Size = 13;
        public const int Border = 2;

        public Cell[,] Cells = new Cell[Size, Size];

        public int Player1Score = 0;
        public int Player2Score = 0;

        public byte currentPlayer = 1;

        public int MoveCount = 0;
        public const int MaxMoves = 80;

        public Board()
        {
            for (int i = 0; i < Size; i++)
                for (int j = 0; j < Size; j++)
                    Cells[i, j] = new Cell();

            // создаём границы
            for (int x = 0; x < Size; x++)
            {
                for (int b = 0; b < Border; b++)
                {
                    MakeMove(x, b, ShapeTemplates.Border, 3);
                    MakeMove(x, Size - 1 - b, ShapeTemplates.Border, 3);
                }
            }

            for (int y = Border; y < Size - Border; y++)
            {
                for (int b = 0; b < Border; b++)
                {
                    MakeMove(b, y, ShapeTemplates.Border, 3);
                    MakeMove(Size - 1 - b, y, ShapeTemplates.Border, 3);
                }
            }
        }

        public bool TryMakeMove(int x, int y, ShapeType type)
        {
            if (MoveCount >= MaxMoves) return false;
            if (Cells[x, y].CurrentShape != null) return false;
            if (x < Border || x >= Size - Border || y < Border || y >= Size - Border)
                return false;

            MakeMove(x, y, type, currentPlayer);
            MoveCount++;

            currentPlayer = (byte)(3 - currentPlayer);
            return true;
        }

        public void MakeMove(int x, int y, ShapeType type, byte player)
        {
            if (Cells[x, y].CurrentShape == null)
            {
                Cells[x, y].CurrentShape = new Shape
                {
                    Type = type,
                    player = player
                };

                if (type != ShapeTemplates.Border)
                {
                    Scores scores = Count(x, y);        
                    Player1Score += scores.Player1;
                    Player2Score += scores.Player2;
                }
            }
        }

        public Scores Count(int x, int y)
        {
            Scores scores = new Scores { Player1 = 0, Player2 = 0 };

            for (int i = x - 1; i < x + 2; ++i)
            {
                for (int j = y - 1; j < y + 2; ++j)
                {
                    int a = 0;
                    int f3 = 0;
                    int f4 = 0;

                    if (i < 0 || i >= Size || j < 0 || j >= Size) continue;

                    if (Cells[i, j].CurrentShape != null)
                    {
                        for (int i1 = i - 1; i1 < i + 2; ++i1)
                        {
                            for (int j1 = j - 1; j1 < j + 2; ++j1)
                            {
                                if (i1 < 0 || i1 >= Size || j1 < 0 || j1 >= Size) continue;

                                if (Cells[i1, j1].CurrentShape != null && !(i1 == i && j1 == j))
                                {
                                    a++;

                                    switch (Cells[i1, j1].CurrentShape.Type.BaseType)
                                    {
                                        case 2: f4++; break;
                                        case 1: f3++; break;
                                    }
                                }
                            }
                        }

                        if (a == 8)
                        {
                            switch (Cells[i, j].CurrentShape.Type.BaseType)
                            {
                                case 2:
                                    if (f4 == 4)
                                    {
                                        if (Cells[i, j].CurrentShape.player == 1) scores.Player1 += 1;
                                        if (Cells[i, j].CurrentShape.player == 2) scores.Player2 += 1;
                                    }
                                    break;

                                case 1:
                                    if (f3 == 3)
                                    {
                                        if (Cells[i, j].CurrentShape.player == 1) scores.Player1 += 1;
                                        if (Cells[i, j].CurrentShape.player == 2) scores.Player2 += 1;
                                    }
                                    break;
                            }
                        }
                    }
                }
            }

            return scores;
        }
    }
}