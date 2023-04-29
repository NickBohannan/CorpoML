using CorpoML.Console.Models;
using CorpoML.Console.Repositores;
using Microsoft.ML;
using Microsoft.ML.Data;


public class RegressionRepository
{
	public EstimatorChain<RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastTreeTweedieModelParameters>>? CreateRegressionPipeline(MLContext mlContext)
	{
        var trainer = mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");

        var trainingPipeline = mlContext.Transforms.Concatenate(outputColumnName: "NumFeatures",
            nameof(ProductData.year), nameof(ProductData.month), nameof(ProductData.units),
            nameof(ProductData.avg), nameof(ProductData.count), nameof(ProductData.max),
            nameof(ProductData.min), nameof(ProductData.prev))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CatFeatures", inputColumnName: nameof(ProductData.productId)))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "NumFeatures", "CatFeatures"))
                .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(ProductData.next)))
                .Append(trainer);

        return trainingPipeline;
    }

    public IReadOnlyList<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> EvaluateRegressionModel(IDataView trainingDataView, MLContext mlContext, EstimatorChain<RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastTreeTweedieModelParameters>> trainingPipeline,
        Microsoft.ML.Trainers.FastTree.FastTreeTweedieTrainer trainer)
    {
        var crossValidationResults = mlContext.Regression.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "Label");

        MLConsoleHelper.PrintRegressionFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

        return crossValidationResults;
    }

    public TransformerChain<RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastTreeTweedieModelParameters>> TrainRegressionModel(EstimatorChain<RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastTreeTweedieModelParameters>> trainingPipeline,
        IDataView trainingDataView)
    {
        return trainingPipeline.Fit(trainingDataView);
    }

    public void SaveRegressionModel(MLContext mlContext, TransformerChain<RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastTreeTweedieModelParameters>> model, IDataView trainingDataView, string outputModelPath)
    {
        using (var file = File.OpenWrite(outputModelPath))
            mlContext.Model.Save(model, trainingDataView.Schema, file);
    }

    public void TestPrediction(string outputModelPath, MLContext mlContext)
    {

        ITransformer trainedModel;
        using (var stream = File.OpenRead(outputModelPath))
        {
            trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
        }

        var predictionEngine = mlContext.Model.CreatePredictionEngine<ProductData, ProductUnitRegressionPrediction>(trainedModel);

        Console.WriteLine("** Testing Product **");

        ProductUnitRegressionPrediction prediction = predictionEngine.Predict(SampleProductData.MonthlyData[0]);
        Console.WriteLine($"Product: {SampleProductData.MonthlyData[0].productId}, month: {SampleProductData.MonthlyData[0].month + 1}, year: {SampleProductData.MonthlyData[0].year} - Real value (units): {SampleProductData.MonthlyData[0].next}, Forecast Prediction (units): {prediction.Score}");

        prediction = predictionEngine.Predict(SampleProductData.MonthlyData[1]);
        Console.WriteLine($"Product: {SampleProductData.MonthlyData[1].productId}, month: {SampleProductData.MonthlyData[1].month + 1}, year: {SampleProductData.MonthlyData[1].year} - Forecast Prediction (units): {prediction.Score}");
    }
}

