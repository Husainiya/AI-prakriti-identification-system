def adaptive_questionnaire(questions):
    scores = {"Vata": 0.0, "Pitta": 0.0, "Kapha": 0.0}
    asked_questions = set()
    asked = 0
    max_questions = 10

    print("\n--- Adaptive Questionnaire ---")

    # -----------------------------
    # STEP 1: Starter Questions
    # -----------------------------
    for dosha in ["Vata", "Pitta", "Kapha"]:
        item = questions[dosha][0]
        q = item["q"]
        weight = item["weight"]

        ans = input(q + " (yes/no): ").strip().lower()

        if ans == "yes":
            scores[dosha] += weight
        else:
            scores[dosha] += 0.5

        asked_questions.add(q)
        asked += 1

    # -----------------------------
    # STEP 2: Adaptive Loop
    # -----------------------------
    while asked < max_questions:

        total = sum(scores.values())

        if total > 0:
            top_dosha = max(scores, key=scores.get)
            confidence = scores[top_dosha] / total
        else:
            confidence = 0

        print(f"Current Scores: {scores}, Confidence: {round(confidence,2)}")

        # Early stop
        if asked >= 6 and confidence >= 0.6:
            print("High confidence reached. Stopping early.")
            break

        # Remaining questions
        remaining = [
            item for item in questions[top_dosha]
            if item["q"] not in asked_questions
        ]

        if not remaining:
            break

        item = remaining[0]
        q = item["q"]
        weight = item["weight"]

        ans = input(q + " (yes/no): ").strip().lower()

        if ans == "yes":
            scores[top_dosha] += weight
        else:
            scores[top_dosha] += 0.5

        asked_questions.add(q)
        asked += 1

    print("\nFinal Questionnaire Scores:", scores)
    return scores