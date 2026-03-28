from matplotlib import pyplot as plt
from train import read_csv
from predict import load_thetas, estimate_price


def show_scatter(km, price):
    t0, t1 = load_thetas("src/model.json")
    prices_estimate = [estimate_price(t0, t1, elem) for elem in km]
    plt.scatter(km, price)
    plt.plot(km, prices_estimate)
    plt.show()


def main():
    km, price = read_csv("src/data.csv")
    show_scatter(km, price)


if __name__ == "__main__":
    main()
