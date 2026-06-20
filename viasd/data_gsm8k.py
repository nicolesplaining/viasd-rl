"""GSM8K loading and prompt construction (chat-templated, few-shot)."""
import torch

SYSTEM = ("You are a helpful assistant that solves grade-school math word problems. "
          "Reason step by step, then state the final answer on its own line as "
          "'The answer is <number>.'")

FEWSHOT = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?",
     "In April she sold 48 clips. In May she sold 48 / 2 = 24 clips. "
     "Altogether she sold 48 + 24 = 72 clips.\nThe answer is 72."),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
     "of babysitting. How much did she earn?",
     "Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned "
     "50 * 0.2 = $10.\nThe answer is 10."),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half "
     "of the money she needs. Her parents decided to give her $15, and her "
     "grandparents twice as much as her parents. How much more money does Betty "
     "need to buy the wallet?",
     "Betty has 100 / 2 = $50. Her grandparents gave 2 * 15 = $30. Now she has "
     "50 + 15 + 30 = $95. She still needs 100 - 95 = $5.\nThe answer is 5."),
    ("James writes a 3-page letter to 2 different friends twice a week. How many "
     "pages does he write a year?",
     "Each time he writes 3 * 2 = 6 pages. Twice a week that is 6 * 2 = 12 pages. "
     "In a year that is 12 * 52 = 624 pages.\nThe answer is 624."),
]


def build_prompt_ids(tokenizer, question, device):
    messages = [{"role": "system", "content": SYSTEM}]
    for q, a in FEWSHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def extract_gold(answer_field: str) -> str:
    # GSM8K gold answers end with "#### <number>"
    return answer_field.split("####")[-1].strip().replace(",", "")


def load_gsm8k(n, split="test"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    n = min(n, len(ds))
    return [(ds[i]["question"], extract_gold(ds[i]["answer"])) for i in range(n)]
