using CorpoML.Console.Models;
using Microsoft.ML;
using CorpoML.Console.Repositories;

MLContext mlContext = new MLContext();

var dataPath = ""; // TODO

var trainingDataView = mlContext.Data.LoadFromTextFile<ProductData>(dataPath, hasHeader: true, separatorChar: ',');

Console.WriteLine("Hello, World!");

