using CorpoML.Console.Models;
using Microsoft.ML;

MLContext mlContext = new MLContext();

var dataPath = ""; // TODO

var trainingDataView = mlContext.Data.LoadFromTextFile<ProductData>(dataPath, hasHeader: true, separatorChar: ',');

Console.WriteLine("Hello, World!");

