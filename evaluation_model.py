import json, time
from query_generator import load_llm, process_query

def evaluate_model():
    llm = load_llm()
    with open("evaluation_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    correct_intent = 0
    valid_sql = 0
    total_time = 0.0

    for sample in dataset:
        q = sample["query"]
        expected_type = sample["expected_type"]
        expected_keywords = sample["expected_sql_keywords"]

        start = time.time()
        result = process_query(q, llm)
        duration = time.time() - start
        total_time += duration

        # Vérifie type
        if result["type"] == expected_type:
            correct_intent += 1

        # Vérifie SQL s’il y en a un
        sql = result.get("sql", "")
        if sql and all(k.lower() in sql.lower() for k in expected_keywords):
            valid_sql += 1

        print(f"\n🧠 Query: {q}")
        print(f"→ Predicted type: {result['type']}, Expected: {expected_type}")
        print(f"→ SQL: {sql}")
        print(f"⏱ Temps de réponse: {duration:.2f}s")

    print("\n===== Résumé =====")
    print(f"Intent Accuracy: {correct_intent/total*100:.1f}%")
    print(f"SQL Accuracy: {valid_sql/total*100:.1f}%")
    print(f"Temps moyen de réponse: {total_time/total:.2f}s")

if __name__ == "__main__":
    evaluate_model()
