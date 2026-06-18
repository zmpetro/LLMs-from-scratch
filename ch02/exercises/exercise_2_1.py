import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Akwirw ier"

integers = tokenizer.encode(text)
print(integers)

for i in integers:
    print(f"{i} -> {tokenizer.decode([i])}")
