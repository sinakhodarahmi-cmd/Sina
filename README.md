#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// A very simple Perceptron model with sigmoid activation
struct Perceptron {
    vector<double> w; // weights
    double bias;
    double lr; // learning rate

    Perceptron(int n_features, double lr_=0.1) {
        w.resize(n_features, 0.0);
        bias = 0.0;
        lr = lr_;
    }

    // Sigmoid activation function
    static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Forward pass: prediction (returns probability in [0,1])
    double predict(const vector<double>& x) {
        double s = bias;
        for (size_t i = 0; i < w.size(); i++)
            s += w[i] * x[i];
        return sigmoid(s);
    }

    // Update weights with gradient descent (online learning)
    void update(const vector<double>& x, int y) {
        double pred = predict(x);
        double error = y - pred;
        for (size_t i = 0; i < w.size(); i++)
            w[i] += lr * error * x[i];
        bias += lr * error;
    }
};

int main() {
    Perceptron p(2, 0.1); // model with 2 features

    // Training dataset: simple AND logic
    vector<vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<int> Y = {0, 0, 0, 1}; // only (1,1) -> 1

    // Training loop
    for (int epoch = 0; epoch < 2000; epoch++) {
        for (size_t i = 0; i < X.size(); i++)
            p.update(X[i], Y[i]);
    }

    cout << "Training finished!\n";
    cout << "Enter new values (x1 x2):\n";

    double a, b;
    while (cin >> a >> b) {
        vector<double> sample = {a, b};
        double prob = p.predict(sample);
        int cls = (prob >= 0.5 ? 1 : 0);
        cout << "Predicted class: " << cls
             << " (probability = " << prob << ")\n";
    }

    return 0;
}
