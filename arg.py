import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--layers', nargs='*', required=True)
parser.add_argument('--def')

args = parser.parse_args()
print(args.layers == None)
