using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML;
using static Microsoft.ML.TrainCatalogBase;
using System.Diagnostics;

namespace CorpoML.Console.Repositores
{
    public static class MLConsoleHelper
    {
        public static void PrintPrediction(string prediction)
        {
           System.Console.WriteLine($"*************************************************");
           System.Console.WriteLine($"Predicted : {prediction}");
           System.Console.WriteLine($"*************************************************");
        }

        public static void PrintRegressionPredictionVersusObserved(string predictionCount, string observedCount)
        {
           System.Console.WriteLine($"-------------------------------------------------");
           System.Console.WriteLine($"Predicted : {predictionCount}");
           System.Console.WriteLine($"Actual:     {observedCount}");
           System.Console.WriteLine($"-------------------------------------------------");
        }

        public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
        {
           System.Console.WriteLine($"*************************************************");
           System.Console.WriteLine($"*       Metrics for {name} regression model      ");
           System.Console.WriteLine($"*------------------------------------------------");
           System.Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
           System.Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
           System.Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
           System.Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
           System.Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
           System.Console.WriteLine($"*************************************************");
        }

        public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
           System.Console.WriteLine($"************************************************************");
           System.Console.WriteLine($"*       Metrics for {name} binary classification model      ");
           System.Console.WriteLine($"*-----------------------------------------------------------");
           System.Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
           System.Console.WriteLine($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
           System.Console.WriteLine($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
           System.Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
           System.Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
           System.Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
           System.Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
           System.Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
           System.Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
           System.Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
           System.Console.WriteLine($"************************************************************");
        }

        public static void PrintAnomalyDetectionMetrics(string name, AnomalyDetectionMetrics metrics)
        {
           System.Console.WriteLine($"************************************************************");
           System.Console.WriteLine($"*       Metrics for {name} anomaly detection model      ");
           System.Console.WriteLine($"*-----------------------------------------------------------");
           System.Console.WriteLine($"*       Area Under ROC Curve:                       {metrics.AreaUnderRocCurve:P2}");
           System.Console.WriteLine($"*       Detection rate at false positive count: {metrics.DetectionRateAtFalsePositiveCount}");
           System.Console.WriteLine($"************************************************************");
        }

        public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
           System.Console.WriteLine($"************************************************************");
           System.Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
           System.Console.WriteLine($"*-----------------------------------------------------------");
           System.Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
           System.Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
           System.Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
           System.Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
           System.Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
           System.Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
           System.Console.WriteLine($"************************************************************");
        }

        public static void PrintRegressionFoldsAverageMetrics(string algorithmName, IReadOnlyList<CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            System.Console.WriteLine($"*************************************************************************************************************");
            System.Console.WriteLine($"*       Metrics for {algorithmName} Regression model      ");
            System.Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            System.Console.WriteLine($"*       Average L1 Loss:    {L1.Average():0.###} ");
            System.Console.WriteLine($"*       Average L2 Loss:    {L2.Average():0.###}  ");
            System.Console.WriteLine($"*       Average RMS:          {RMS.Average():0.###}  ");
            System.Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            System.Console.WriteLine($"*       Average R-squared: {R2.Average():0.###}  ");
            System.Console.WriteLine($"*************************************************************************************************************");
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(
                                            string algorithmName,
                                        IReadOnlyList<CrossValidationResult<MulticlassClassificationMetrics>> crossValResults
                                                                            )
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            System.Console.WriteLine($"*************************************************************************************************************");
            System.Console.WriteLine($"*       Metrics for {algorithmName} Multi-class Classification model      ");
            System.Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            System.Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            System.Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            System.Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            System.Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            System.Console.WriteLine($"*************************************************************************************************************");

        }

        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }

        public static void PrintClusteringMetrics(string name, ClusteringMetrics metrics)
        {
            System.Console.WriteLine($"*************************************************");
            System.Console.WriteLine($"*       Metrics for {name} clustering model      ");
            System.Console.WriteLine($"*------------------------------------------------");
            System.Console.WriteLine($"*       Average Distance: {metrics.AverageDistance}");
            System.Console.WriteLine($"*       Davies Bouldin Index is: {metrics.DaviesBouldinIndex}");
            System.Console.WriteLine($"*************************************************");
        }

        public static void ShowDataViewInConsole(MLContext mlContext, IDataView dataView, int numberOfRows = 4)
        {
            string msg = string.Format("Show data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            var preViewTransformedData = dataView.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                System.Console.WriteLine(lineToPrint + "\n");
            }
        }

        [Conditional("DEBUG")]
        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        public static void PeekDataViewInConsole(MLContext mlContext, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            //https://github.com/dotnet/machinelearning/blob/main/docs/code/MlNetCookBook.md#how-do-i-look-at-the-intermediate-data
            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // 'transformedData' is a 'promise' of data, lazy-loading. call Preview
            //and iterate through the returned collection from preview.

            var preViewTransformedData = transformedData.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                System.Console.WriteLine(lineToPrint + "\n");
            }
        }

        [Conditional("DEBUG")]
        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        public static void PeekVectorColumnDataInConsole(MLContext mlContext, string columnName, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: : Show {0} rows with just the '{1}' column", numberOfRows, columnName);
            ConsoleWriteHeader(msg);

            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // Extract the 'Features' column.
            var someColumnData = transformedData.GetColumn<float[]>(columnName)
                                                        .Take(numberOfRows).ToList();

            // print to console the peeked rows

            int currentRow = 0;
            someColumnData.ForEach(row =>
            {
                currentRow++;
                String concatColumn = String.Empty;
                foreach (float f in row)
                {
                    concatColumn += f.ToString();
                }

                System.Console.WriteLine();
                string rowMsg = string.Format("**** Row {0} with '{1}' field value ****", currentRow, columnName);
                System.Console.WriteLine(rowMsg);
                System.Console.WriteLine(concatColumn);
                System.Console.WriteLine();
            });
        }

        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = System.Console.ForegroundColor;
            System.Console.ForegroundColor = ConsoleColor.Yellow;
            System.Console.WriteLine(" ");
            foreach (var line in lines)
            {
                System.Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            System.Console.WriteLine(new string('#', maxLength));
            System.Console.ForegroundColor = defaultColor;
        }

        public static void ConsoleWriterSection(params string[] lines)
        {
            var defaultColor = System.Console.ForegroundColor;
            System.Console.ForegroundColor = ConsoleColor.Blue;
            System.Console.WriteLine(" ");
            foreach (var line in lines)
            {
                System.Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            System.Console.WriteLine(new string('-', maxLength));
            System.Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = System.Console.ForegroundColor;
            System.Console.ForegroundColor = ConsoleColor.Green;
            System.Console.WriteLine(" ");
            System.Console.WriteLine("Press any key to finish.");
            System.Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = System.Console.ForegroundColor;
            System.Console.ForegroundColor = ConsoleColor.Red;
            const string exceptionTitle = "EXCEPTION";
            System.Console.WriteLine(" ");
            System.Console.WriteLine(exceptionTitle);
            System.Console.WriteLine(new string('#', exceptionTitle.Length));
            System.Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                System.Console.WriteLine(line);
            }
        }

        public static void ConsoleWriteWarning(params string[] lines)
        {
            var defaultColor = System.Console.ForegroundColor;
            System.Console.ForegroundColor = ConsoleColor.DarkMagenta;
            const string warningTitle = "WARNING";
            System.Console.WriteLine(" ");
            System.Console.WriteLine(warningTitle);
            System.Console.WriteLine(new string('#', warningTitle.Length));
            System.Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                System.Console.WriteLine(line);
            }
        }
    }
}

