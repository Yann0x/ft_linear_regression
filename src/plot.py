import matplotlib
matplotlib.use("qtagg")
from matplotlib import pyplot as plt
from train import read_csv
from predict import load_thetas, estimate_price
import sys


def show_scatter(km, price, model_file="model.json"):
    t0, t1 = load_thetas(model_file)
    prices_estimate = [estimate_price(t0, t1, elem) for elem in km]
    plt.scatter(km, price, label="Data")
    plt.plot(km, prices_estimate, color="red", label="Regression line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot.py <dataset.csv> [model.json]")
        exit(1)
    dataset = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "model.json"
    km, price = read_csv(dataset)
    show_scatter(km, price, model_file)


if __name__ == "__main__":
    main()
