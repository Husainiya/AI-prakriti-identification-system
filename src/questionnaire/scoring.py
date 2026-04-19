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
    # STEP 2: Adaptive Loop (IMPROVED)
    # -----------------------------
    while asked < max_questions:

        total = sum(scores.values())

        if total > 0:
            # Sort doshas by score
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            confidence = sorted_scores[0][1] / total
        else:
            confidence = 0
            sorted_scores = [("Vata",0), ("Pitta",0), ("Kapha",0)]

        print(f"Current Scores: {scores}, Confidence: {round(confidence,2)}")

        # -----------------------------
        # Early Stop
        # -----------------------------
        if asked >= 6 and confidence >= 0.6:
            print("High confidence reached. Stopping early.")
            break

        # -----------------------------
        # SMART DOSHA SELECTION
        # -----------------------------
        top = sorted_scores[0]
        second = sorted_scores[1]

        # If scores are close → alternate
        if abs(top[1] - second[1]) < 1:
            if asked % 2 == 0:
                chosen_dosha = top[0]
            else:
                chosen_dosha = second[0]
        else:
            chosen_dosha = top[0]

        # -----------------------------
        # Get remaining questions
        # -----------------------------
        remaining = [
            item for item in questions[chosen_dosha]
            if item["q"] not in asked_questions
        ]

        # If no questions left → try next dosha
        if not remaining:
            for d in ["Vata", "Pitta", "Kapha"]:
                alt = [
                    item for item in questions[d]
                    if item["q"] not in asked_questions
                ]
                if alt:
                    chosen_dosha = d
                    remaining = alt
                    break

        if not remaining:
            print("No more questions available.")
            break

        # Ask question
        item = remaining[0]
        q = item["q"]
        weight = item["weight"]

        ans = input(q + " (yes/no): ").strip().lower()

        if ans == "yes":
            scores[chosen_dosha] += weight
        else:
            scores[chosen_dosha] += 0.5

        asked_questions.add(q)
        asked += 1

    print("\nFinal Questionnaire Scores:", scores)
    return scores