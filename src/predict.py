import json
import sys


def estimate_price(theta0, theta1, km):
    return theta0 + (theta1 * km)


def load_thetas(filename="model.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0, 0
    except json.JSONDecodeError:
        print(f"Error: '{filename}' is corrupted")
        exit(1)
    if "t0" not in data or "t1" not in data:
        print(f"Error: '{filename}' is missing 't0' or 't1'")
        exit(1)
    return data["t0"], data["t1"]


def main():
    model_file = sys.argv[1] if len(sys.argv) > 1 else "model.json"
    try:
        mileage = int(input("Please enter a mileage: "))
    except ValueError:
        print("Error: mileage must be a number")
        return
    if mileage < 0:
        print("Error: mileage cannot be negative")
        return
    t0, t1 = load_thetas(model_file)
    price = round(estimate_price(t0, t1, mileage))
    if price < 0:
        print("This car is not worth selling")
    else:
        print(f"The estimated price of your car is : {price}")


if __name__ == "__main__":
    main()
