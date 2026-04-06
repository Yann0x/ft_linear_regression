from predict import load_thetas, estimate_price
from train import read_csv
from statistics import mean
import sys


def compute_R_squared(dataset, model_file="model.json"):
    my_model_sum = 0
    null_model_sum = 0
    km, price = read_csv(dataset)
    t0, t1 = load_thetas(model_file)
    price_mean = mean(price)
    for i in range(len(km)):
        my_model_sum += (estimate_price(t0, t1, km[i]) - price[i]) ** 2
        null_model_sum += (price_mean - price[i]) ** 2
    if null_model_sum == 0:
        print("Error: all prices are identical, R Squared is undefined")
        exit(1)
    return 1 - my_model_sum / null_model_sum


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 precision.py <dataset.csv> [model.json]")
        exit(1)
    dataset = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "model.json"
    print(f"The model is {round(compute_R_squared(dataset, model_file), 4) * 100}% precise!")


if __name__ == "__main__":
    main()
