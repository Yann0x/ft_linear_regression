from predict import load_thetas, estimate_price
from train import read_csv
from statistics import mean


def compute_R_squared():
    my_model_sum = 0
    null_model_sum = 0
    km, price = read_csv("src/data.csv")
    t0, t1 = load_thetas("src/model.json")
    price_mean = mean(price)
    for i in range(len(km)):
        my_model_sum += (estimate_price(t0, t1, km[i]) - price[i]) ** 2
        null_model_sum += (price_mean - price[i]) ** 2
    if null_model_sum == 0:
        print("Error: all prices are identical, R² is undefined")
        exit(1)
    return 1 - my_model_sum / null_model_sum


def main():
    print(f"The model is {round(compute_R_squared(), 4) * 100}% precise!")


if __name__ == "__main__":
    main()
