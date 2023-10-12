# test colorize eccv16
from colorization.colorizers.eccv16 import eccv16

def run():
    print("Loading model")
    model = eccv16().eval()
    print("Loading complete!")

# test colorize coltran

if __name__ == '__main__':
    run()