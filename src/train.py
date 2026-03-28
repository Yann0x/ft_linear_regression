import pandas as pd
from predict import estimate_price
import json


def read_csv(filename):
    try:
        data = pd.DataFrame(pd.read_csv(filename))
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found")
        exit(1)
    if "km" not in data.columns or "price" not in data.columns:
        print("Error: CSV must contain 'km' and 'price' columns")
        exit(1)
    km = data["km"]
    price = data["price"]
    return (km.tolist(), price.tolist())


def normalize(data):
    nmax = max(data)
    nmin = min(data)
    if nmax == nmin:
        print("Error: cannot normalize data with all identical values")
        exit(1)
    data = [(element - nmin) / (nmax - nmin) for element in data]
    return data


def gradient_descent(km, price):
    t0, t1 = 0, 0
    epochs = 999
    learning_rate = 0.1
    for i in range(0, epochs):
        sum0, sum1 = 0, 0
        for j in range(0, len(km)):
            sum0 += estimate_price(t0, t1, km[j]) - price[j]
            sum1 += (estimate_price(t0, t1, km[j]) - price[j]) * km[j]
        tmpt0 = learning_rate * (1 / len(km)) * sum0
        tmpt1 = learning_rate * (1 / len(km)) * sum1
        t0 -= tmpt0
        t1 -= tmpt1
    return (t0, t1)


def denormalize_ts(t0, t1, km, price):
    max_price = max(price)
    min_price = min(price)
    max_km = max(km)
    min_km = min(km)
    t1_reel = t1 * (max_price - min_price) / (max_km - min_km)
    t0_reel = t0 * (max_price - min_price) + min_price - t1_reel * min_km
    return (t0_reel, t1_reel)


def save_ts(t0, t1):
    data = {"t0": t0, "t1": t1}
    with open("src/model.json", "w") as f:
        json.dump(data, f)


def main():
    km, price = read_csv("src/data.csv")
    km_norm = normalize(km)
    price_norm = normalize(price)
    t0, t1 = gradient_descent(km_norm, price_norm)
    t0_reel, t1_reel = denormalize_ts(t0, t1, km, price)
    save_ts(t0_reel, t1_reel)


if __name__ == "__main__":
    main()
