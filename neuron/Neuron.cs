/*
 * User: IPikin
 * Date: 31-Jul-18
 * Time: 14:52
 */
using System;
using System.Collections.Generic;

namespace NeyronTest.neuron
{
    public class Neuron
    {
        public const float DEFAULT_LEARNING_RATE = 0.1f;
        
        protected Dictionary<Neuron, float> weights;
        protected List<Neuron> inputNeurons;
        protected int id;
        protected float output;
        protected float error;
		
        static readonly Random rnd = new Random();
        static int idCounter;
		
        public Neuron(float learningRate = DEFAULT_LEARNING_RATE)
        {
            this.id = GetNewId();
            this.LearningRate = learningRate;
        }
		
        protected static int GetNewId()
        {
            return idCounter++;
        }
		
        public int Id { get { return id; } }
		
        public float Input  { get; set; }
		
        public float LearningRate { get; set; }
		
        public float Output { get { return output; } }
		
        public int OutputBinary { get { return output >= 0.5 ? 1 : 0; } }
		
        public float Error { get { return Math.Abs(error); } }
		
        public void SetInputNeurons(List<Neuron> inputNeurons)
        {
            weights = new Dictionary<Neuron, float>();
            foreach (Neuron neuron in inputNeurons) {
                weights.Add(neuron, GetRandomWeight());
            }
            this.inputNeurons = inputNeurons;
        }
		
        public float Compute()
        {
            float result = 0;
			
            if (inputNeurons == null) {
                result = Input;
            } else {
                foreach (Neuron neuron in inputNeurons) {
                    result += neuron.Compute() * weights[neuron];
                }
                Input = result;
                result = Sigmoid(result);
            }
			
            output = result;
			
            return result;
        }
		
        public void Learn(float expected)
        {
            LearnNext(output - expected);
        }
		
        protected void LearnNext(float error)
        {
            this.error = error;
			
            float deltaWeight;
            if (inputNeurons != null) {
                foreach (Neuron neuron in inputNeurons) {
                    deltaWeight = error * output * (1 - output);
                    weights[neuron] -= neuron.output * deltaWeight * LearningRate;
                    neuron.LearnNext(weights[neuron] * deltaWeight);
                }
            }
        }

        protected float Sigmoid(float value)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-value));
        }

        float GetRandomWeight()
        {
            return (float)rnd.Next() / (float)int.MaxValue;
        }
    }
}
