using System;

public class NeuralNetworks
{
    private int inputSize;
    private int hiddenSize;
    private int outputSize;

    private double[,] inputWeights;
    private double[] inputBiases;

    private double[,] outputWeights;
    private double[] outputBiases;

    public NeuralNetworks(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Initialize input layer weights and biases
        inputWeights = new double[inputSize, hiddenSize];
        inputBiases = new double[hiddenSize];

        Random rand = new Random();
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                inputWeights[i, j] = rand.NextDouble() - 0.5;
            }
        }

        for (int j = 0; j < hiddenSize; j++)
        {
            inputBiases[j] = rand.NextDouble() - 0.5;
        }

        // Initialize output layer weights and biases
        outputWeights = new double[hiddenSize, outputSize];
        outputBiases = new double[outputSize];

        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                outputWeights[i, j] = rand.NextDouble() - 0.5;
            }
        }

        for (int j = 0; j < outputSize; j++)
        {
            outputBiases[j] = rand.NextDouble() - 0.5;
        }
    }

    private double[] Sigmoid(double[] x)
    {
        double[] y = new double[x.Length];

        for (int i = 0; i < x.Length; i++)
        {
            y[i] = 1 / (1 + Math.Exp(-x[i]));
        }

        return y;
    }

    public double[] FeedForward(double[] input)
    {
        double[] hidden = new double[hiddenSize];

        // Calculate input layer activation
        for (int j = 0; j < hiddenSize; j++)
        {
            double sum = 0;

            for (int i = 0; i < inputSize; i++)
            {
                sum += input[i] * inputWeights[i, j];
            }

            sum += inputBiases[j];
            hidden[j] = sum;
        }

        // Apply sigmoid activation to hidden layer
        hidden = Sigmoid(hidden);

        double[] output = new double[outputSize];

        // Calculate output layer activation
        for (int j = 0; j < outputSize; j++)
        {
            double sum = 0;

            for (int i = 0; i < hiddenSize; i++)
            {
                sum += hidden[i] * outputWeights[i, j];
            }

            sum += outputBiases[j];
            output[j] = sum;
        }

        // Apply sigmoid activation to output layer
        output = Sigmoid(output);

        return output;
    }

    public void Train(double[] input, double[] targetOutput, double learningRate)
    {
        // Calculate activations
        double[] hidden = new double[hiddenSize];

        // Calculate input layer activation
        for (int j = 0; j < hiddenSize; j++)
        {
            double sum = 0;

            for (int i = 0; i < inputSize; i++)

            {
                sum += input[i] * inputWeights[i, j];
            }
            sum += inputBiases[j];
            hidden[j] = sum;
        }
        // Apply sigmoid activation to hidden layer
        hidden = Sigmoid(hidden);

        double[] output = new double[outputSize];

        // Calculate output layer activation
        for (int j = 0; j < outputSize; j++)
        {
            double sum = 0;

            for (int i = 0; i < hiddenSize; i++)
            {
                sum += hidden[i] * outputWeights[i, j];
            }

            sum += outputBiases[j];
            output[j] = sum;
        }

        // Apply sigmoid activation to output layer
        output = Sigmoid(output);

        // Calculate error
        double[] outputError = new double[outputSize];

        for (int j = 0; j < outputSize; j++)
        {
            outputError[j] = (targetOutput[j] - output[j]) * output[j] * (1 - output[j]);
        }

        double[] hiddenError = new double[hiddenSize];

        for (int j = 0; j < hiddenSize; j++)
        {
            double sum = 0;

            for (int i = 0; i < outputSize; i++)
            {
                sum += outputError[i] * outputWeights[j, i];
            }

            hiddenError[j] = sum * hidden[j] * (1 - hidden[j]);
        }

        // Update output layer weights and biases
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                outputWeights[i, j] += learningRate * outputError[j] * hidden[i];
            }
        }

        for (int j = 0; j < outputSize; j++)
        {
            outputBiases[j] += learningRate * outputError[j];
        }

        // Update input layer weights and biases
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                inputWeights[i, j] += learningRate * hiddenError[j] * input[i];
            }
        }

        for (int j = 0; j < hiddenSize; j++)
        {
            inputBiases[j] += learningRate * hiddenError[j];
        }
    }
}
