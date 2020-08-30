/*
 * Created by SharpDevelop.
 * User: IPikin
 * Date: 02-Aug-18
 * Time: 15:27
 */
using System;
using System.Collections.Generic;

namespace NeyronTest.lib
{
	public class NeuronsLayer
	{
		protected int id;
		protected List<Neuron> neurons;
		protected NeuronsLayer inputNeuronsLayer;
		
		static int idCounter;
		
		public NeuronsLayer(int neuronsCount, float learningRate)
		{
			id = GetNewId();
			
			neurons = new List<Neuron>(neuronsCount);
			while (neuronsCount-- > 0) neurons.Add(new Neuron(learningRate));
		}
		
		protected static int GetNewId() {
			return idCounter++;
		}
		
		public int Id { get { return id; } }
		
		public List<Neuron> Neurons {
			get { return neurons; }
		}
		
		public void SetInputNeuronsLayer(NeuronsLayer inputNeuronsLayer)
		{
			this.inputNeuronsLayer = inputNeuronsLayer;
			foreach (Neuron neuron in neurons)
				neuron.SetInputNeurons(inputNeuronsLayer.Neurons);
		}
		
		public void Compute()
		{
			foreach (Neuron neuron in neurons)
				neuron.Compute();
		}
		
		public void Learn(float[] expectedData)
		{
			Compute();
			for (int i = 0 ; i < neurons.Count; i++)
				neurons[i].Learn(expectedData[i]);
		}
		
		public void SetInputData(float[] inputData)
		{
			for (int i = 0 ; i < neurons.Count; i++)
				neurons[i].Input = inputData[i];
		}
	}
}
