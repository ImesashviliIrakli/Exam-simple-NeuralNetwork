namespace NeuralNetwork
{
    public class Program
    {
        static void Main(string[] args)
        {
            // Create a neural network with 2 inputs, 3 hidden neurons, and 1 output
            NeuralNetworks net = new NeuralNetworks(2, 3, 1);

            // Train the network on a set of inputs and target outputs
            double[,] inputs = { { 1, 0 }, { 0, 1 }, { 1, 1 }, { 0, 0 } };
            double[,] targets = { { 1 }, { 1 }, { 0 }, { 0 } };

            double learningRate = 0.2;

            for (int i = 0; i < 10000; i++)
            {
                for (int j = 0; j < inputs.GetLength(0); j++)
                {
                    net.Train(new double[] { inputs[j, 0], inputs[j, 1] }, new double[] { targets[j, 0] }, learningRate);
                }
            }

            // Test the trained network
            double[] output = net.FeedForward(new double[] { 1, 0 });
            Console.WriteLine(output[0]); // Expected output: close to 1

            output = net.FeedForward(new double[] { 0, 1 });
            Console.WriteLine(output[0]); // Expected output: close to 1

            output = net.FeedForward(new double[] { 1, 1 });
            Console.WriteLine(output[0]); // Expected output: close to 0

            output = net.FeedForward(new double[] { 0, 0 });
            Console.WriteLine(output[0]); // Expected output: close to 0

            Console.ReadKey();
        }
    }
}