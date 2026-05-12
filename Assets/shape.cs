using System;

namespace ConsoleGameLibrary
{
    public class ShapeType
    {
        public string name;
        public byte BaseType;     // 1 - треугольник, 2 - квадрат, 5 - граница
        public byte CanUp;
    }

    public class Shape
    {
        public ShapeType Type;
        public byte player;       // 1 или 2 - игроки, 3 - граница
    }

    public class ShapeTemplates
    {
        public static ShapeType Cube = new ShapeType 
        { 
            name = "Cube", 
            BaseType = 2, 
            CanUp = 0 
        };

        public static ShapeType TrianglePrism = new ShapeType 
        { 
            name = "TrianglePrism", 
            BaseType = 1, 
            CanUp = 0 
        };

        public static ShapeType Border = new ShapeType 
        { 
            name = "Border", 
            BaseType = 5, 
            CanUp = 2 
        };
    }
}