/*
 * Created by SharpDevelop.
 * User: IPikin
 * Date: 31-Jul-18
 * Time: 14:30
 * 
 * To change this template use Tools | Options | Coding | Edit Standard Headers.
 */
using System;
using System.Collections.Generic;
using System.Linq;
using NeyronTest.lib;

namespace NeyronTest
{
	class Program
	{
		const double LEARN_COUNT = 1e5;
		const float LEARNING_RATE = 0.1f;
		
		const int INPUT_NEURONS_COUNT = 3;
		const int HIDDEN_NEURONS_COUNT = 8;
		const int OUTPUT_NEURONS_COUNT = 2;
		const int TRACE_LAST_LEARN_ITERATIONS = 10;
		
		const string DECIMAL_FORMAT = "0.######";
		
		static readonly NeuronsLayer inputNeurons = new NeuronsLayer(INPUT_NEURONS_COUNT, LEARNING_RATE);
		static readonly NeuronsLayer hiddenNeurons = new NeuronsLayer(HIDDEN_NEURONS_COUNT, LEARNING_RATE);
		static readonly NeuronsLayer outputNeurons = new NeuronsLayer(OUTPUT_NEURONS_COUNT, LEARNING_RATE);
		
		static readonly Dictionary<float[], float[]> learnDataInOutMap = new Dictionary<float[], float[]>() {
			{ new float[]{0, 0, 0}, new float[]{1, 0} },
			{ new float[]{0, 0, 1}, new float[]{0, 0} },
			{ new float[]{0, 1, 1}, new float[]{0, 0} },
			{ new float[]{1, 1, 1}, new float[]{1, 1} },
		};
		
		static readonly float[][] testData = {
			new float[]{1, 0, 1},
			new float[]{1, 0, 0}
		};
		
		
		public static void Main(string[] args)
		{
			InInitialize();
			RunLearning();
			DoTests();
			
			Console.WriteLine();
			Console.Write("Press any key to continue . . . ");
			Console.ReadKey(true);
		}
		
		static void InInitialize()
		{
			hiddenNeurons.SetInputNeuronsLayer(inputNeurons);
			outputNeurons.SetInputNeuronsLayer(hiddenNeurons);
		}
		
		static void RunLearning()
		{
			Console.WriteLine("Start Learning . . . ");
			
			double learnCount = 0;
			while (learnCount++ < LEARN_COUNT) {
				if (learnCount >= LEARN_COUNT - TRACE_LAST_LEARN_ITERATIONS) {
					Console.WriteLine();
					Console.WriteLine("Learning iteration# " + learnCount);
				}
				
				foreach (KeyValuePair<float[], float[]> entry in learnDataInOutMap) {
					inputNeurons.SetInputData(entry.Key);
					outputNeurons.Learn(entry.Value);
					
					if (learnCount >= LEARN_COUNT - TRACE_LAST_LEARN_ITERATIONS) {
						TraceLearnResult(entry.Key, entry.Value);
					}
				}
			}
		}
		
		static void TraceLearnResult(float[] inputData, float[] outputData)
		{
			Console.WriteLine("Input data: [" + String.Join(",", inputData) + "]");
			
			for (int i = 0; i < outputData.GetLength(0); i++) {
				Console.WriteLine(
					"Neuron# " + i +
					" | Expected: " + outputData[i] +
					" | Out: " + outputNeurons.Neurons[i].Output.ToString(DECIMAL_FORMAT) +
					" | Out Bin: " + outputNeurons.Neurons[i].OutputBinary +
					" | Err: " + outputNeurons.Neurons[i].Error.ToString(DECIMAL_FORMAT));
			}
		}
		
		static void DoTests()
		{
			Console.WriteLine("Tests");
			int i = 0;
			
			foreach (var inputData in testData) {
				Console.WriteLine();
				Console.WriteLine("Test# " + i++);
				
				inputNeurons.SetInputData(inputData);
				outputNeurons.Compute();
				
				Console.WriteLine("Input data: [" + String.Join(",", inputData) + "]");
				
				for (int j = 0; j < OUTPUT_NEURONS_COUNT; j++) {
					Console.WriteLine(
						"Neuron# " + j +
						" | Out: " + outputNeurons.Neurons[j].Output.ToString(DECIMAL_FORMAT) +
						" | Out Bin: " + outputNeurons.Neurons[j].OutputBinary);
				}
			}
			
		}
		
	}
}