# ==========================================
# SKINO_AI - Personalized Skin Questionnaire
# ==========================================

from recommendation_engine import get_recommendations


def ask_question(question, options):
    print(f"\n{question}")

    for number, option in enumerate(options, start=1):
        print(f"{number}. {option}")

    while True:
        answer = input("Enter option number: ").strip()

        if answer.isdigit():
            answer = int(answer)

            if 1 <= answer <= len(options):
                return options[answer - 1]

        print("Invalid choice. Please enter a valid option.")


def run_questionnaire():

    print("\n" + "=" * 45)
    print("       SKINO_AI PERSONAL SKIN ANALYSIS")
    print("=" * 45)

    skin_type = ask_question(
        "What is your skin type?",
        ["Dry", "Oily", "Normal", "Combination", "Not sure"]
    )

    acne = ask_question(
        "Do you experience acne or pimples?",
        ["Never", "Occasionally", "Frequently"]
    )

    sensitivity = ask_question(
        "How sensitive is your skin?",
        ["Not sensitive", "Sometimes irritated", "Very sensitive"]
    )

    dryness = ask_question(
        "Do you experience dry or flaky skin?",
        ["Never", "Sometimes", "Frequently"]
    )

    pigmentation = ask_question(
        "Do you have dark spots or uneven skin tone?",
        ["No", "Mild", "Noticeable"]
    )

    print("\n" + "=" * 45)
    print("          YOUR PERSONALIZED REPORT")
    print("=" * 45)

    # Skin type recommendations
    if skin_type == "Not sure":
        print(
            "\nSkin type is not confirmed. "
            "Choose gentle products until you understand your skin."
        )
    else:
        recommendations = get_recommendations(skin_type)

        if recommendations:
            print(f"\nSelected skin type: {skin_type}")

            for category, details in recommendations.items():
                print(f"\n{category.upper()}")
                print("Ingredients:", ", ".join(details["Ingredients"]))
                print("Purpose:", details["Benefits"])

    # Concern-based guidance
    print("\n--- YOUR SKIN CONCERNS ---")

    if acne != "Never":
        print("\nAcne:")
        print("- Consider a gentle, non-comedogenic cleanser.")
        print("- Salicylic acid may help some people with acne.")
        print("- Persistent or severe acne may need a dermatologist.")

    if sensitivity != "Not sensitive":
        print("\nSensitivity:")
        print("- Prefer fragrance-free, gentle products.")
        print("- Avoid introducing several active ingredients together.")
        print("- Stop products that cause persistent irritation.")

    if dryness != "Never":
        print("\nDryness:")
        print("- Look for glycerin and ceramides in moisturizers.")
        print("- Avoid harsh cleansers and excessive exfoliation.")

    if pigmentation != "No":
        print("\nUneven skin tone:")
        print("- Daily broad-spectrum SPF 30+ sunscreen is important.")
        print("- Avoid aggressive exfoliation if your skin is irritated.")
        print("- Persistent pigmentation can be assessed by a dermatologist.")

    print("\n--- BASIC ROUTINE ---")
    print("Morning: Gentle cleanser -> Moisturizer -> Sunscreen")
    print("Night: Gentle cleanser -> Moisturizer")

    print("\nNote:")
    print("This questionnaire provides general skincare guidance.")
    print("It does not diagnose skin diseases.")


if __name__ == "__main__":
    run_questionnaire()