import argparse

def main():
    parser = argparse.ArgumentParser(description="A command-line script with --name argument.")
    parser.add_argument("-n","--name", type=float, help="Your name")

    args = parser.parse_args()

    if args.name:
        print(f"Hello, {args.name}!")

if __name__ == "__main__":
    main()