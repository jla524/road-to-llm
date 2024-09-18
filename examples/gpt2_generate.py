import argparse
import numpy as np
from tinygrad import Tensor, Device, Variable
from tinygrad.helpers import colored
from road_to_llm.models.gpt2 import GPT2, MAX_CONTEXT


Tensor.no_grad = True
print(f"using {Device.DEFAULT} backend")
default_prompt = "What is the answer to life, the universe, and everything?"

parser = argparse.ArgumentParser(description="Run GPT2 in tinygrad")
parser.add_argument("--prompt", type=str, default=default_prompt, help="Phrase to start with")
parser.add_argument("--count", type=int, default=100, help="Max number of tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8, help="Temperature in the softmax")
parser.add_argument("--model_size", type=str, default="gpt2-medium", help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]")
parser.add_argument("--seed", type=int, help="Set the random seed")
parser.add_argument("--batch_size", type=int, default=1, help="Set the input batch size")
parser.add_argument("--benchmark", type=int, default=-1, help="Benchmark GPT with the given number of tokens")
parser.add_argument("--noshow", action="store_true", help="Don't show the output")
args = parser.parse_args()

if args.seed is not None:
    Tensor.manual_seed(args.seed)
    np.random.seed(args.seed)

print(f"using {args.model_size}")
gpt2 = GPT2.build(args.model_size)

if args.benchmark != -1:
    gpt2.model(Tensor.rand(args.batch_size, args.benchmark), Variable("a", 0, MAX_CONTEXT).bind(0)).realize()
else:
    texts = gpt2.generate(args.prompt, args.count, args.temperature, batch_size=args.batch_size)
    assert len(texts) > 0
    if not args.noshow:
        print("Generating text...")
        if len(texts) == 1:
            print(texts[0])
        else:
            for i, text in enumerate(texts):
                print(colored(f"Response {i}: ", "green"), text)

    # validate output!
    if args.temperature == 0 and args.model_size == "gpt2-medium" and args.count == 10:
        expected = {
            default_prompt: f"{default_prompt}\n\nThe answer is that we are all one",
            "Hello.": "Hello. I'm a little late to the party, but",
        }
        if args.prompt in expected:
            assert texts[0] == expected[args.prompt]
            print(colored("output validated", "green"))
