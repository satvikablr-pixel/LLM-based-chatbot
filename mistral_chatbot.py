from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

print("AI Chatbot (type exit to stop)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = "Explain in detail: " + user_input

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(inputs["input_ids"], max_new_tokens=200)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Bot:", answer)
