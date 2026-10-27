# ==========================================
# SKINO_AI - Ingredient-Based Recommendation
# ==========================================

def get_recommendations(skin_type):

    recommendations = {
        "Dry": {
            "Cleanser": {
                "Ingredients": [
                    "Glycerin",
                    "Hyaluronic Acid",
                    "Ceramides"
                ],
                "Benefits": (
                    "Skin ko gently clean karne aur "
                    "moisture retain karne mein help karte hain."
                ),
                "Look For": "Gentle, fragrance-free cleanser"
            },

            "Moisturizer": {
                "Ingredients": [
                    "Ceramides",
                    "Hyaluronic Acid",
                    "Glycerin",
                    "Squalane"
                ],
                "Benefits": (
                    "Skin barrier support karte hain aur "
                    "dryness ko reduce karne mein help karte hain."
                ),
                "Look For": "Cream-based moisturizer"
            },

            "Sunscreen": {
                "Ingredients": [
                    "Zinc Oxide",
                    "Titanium Dioxide",
                    "OR approved organic UV filters"
                ],
                "Benefits": (
                    "UV rays se protection dete hain. "
                    "Broad-spectrum SPF 30+ choose karein."
                ),
                "Look For": "Moisturizing broad-spectrum SPF 30+"
            },

            "Optional Serum": {
                "Ingredients": [
                    "Hyaluronic Acid"
                ],
                "Benefits": (
                    "Skin ko hydrated rakhne mein help karta hai."
                ),
                "Precaution": (
                    "Damp skin par lagayein aur moisturizer use karein."
                )
            }
        },

        "Oily": {
            "Cleanser": {
                "Ingredients": [
                    "Salicylic Acid (0.5-2%)",
                    "Glycerin"
                ],
                "Benefits": (
                    "Salicylic acid pores ke andar oil aur "
                    "buildup ko manage karne mein help kar sakta hai."
                ),
                "Look For": (
                    "Gentle cleanser; salicylic acid optional hai."
                )
            },

            "Moisturizer": {
                "Ingredients": [
                    "Glycerin",
                    "Ceramides",
                    "Hyaluronic Acid"
                ],
                "Benefits": (
                    "Hydration aur skin barrier support dete hain "
                    "bina heavy cream ki zarurat ke."
                ),
                "Look For": "Lightweight, non-comedogenic moisturizer"
            },

            "Sunscreen": {
                "Ingredients": [
                    "Zinc Oxide",
                    "Titanium Dioxide",
                    "OR approved organic UV filters"
                ],
                "Benefits": (
                    "UV protection dete hain. Broad-spectrum "
                    "SPF 30+ aur lightweight texture choose karein."
                ),
                "Look For": "Lightweight, non-comedogenic sunscreen"
            },

            "Optional Serum": {
                "Ingredients": [
                    "Niacinamide (around 2-5%)"
                ],
                "Benefits": (
                    "Oiliness aur uneven skin appearance ko "
                    "manage karne mein help kar sakta hai."
                ),
                "Precaution": (
                    "Low concentration se start karein; irritation "
                    "ho to use band kar dein."
                )
            }
        },

        "Normal": {
            "Cleanser": {
                "Ingredients": [
                    "Glycerin",
                    "Mild cleansing agents"
                ],
                "Benefits": (
                    "Skin ko clean karte hain aur unnecessary "
                    "dryness se bachane mein help karte hain."
                ),
                "Look For": "Gentle cleanser"
            },

            "Moisturizer": {
                "Ingredients": [
                    "Ceramides",
                    "Glycerin",
                    "Hyaluronic Acid"
                ],
                "Benefits": (
                    "Skin barrier aur hydration ko support karte hain."
                ),
                "Look For": "Lightweight moisturizer"
            },

            "Sunscreen": {
                "Ingredients": [
                    "Zinc Oxide",
                    "Titanium Dioxide",
                    "OR approved organic UV filters"
                ],
                "Benefits": (
                    "Broad-spectrum SPF 30+ UV protection "
                    "ke liye choose karein."
                ),
                "Look For": "Broad-spectrum SPF 30+"
            },

            "Optional Serum": {
                "Ingredients": [
                    "Hyaluronic Acid",
                    "Niacinamide"
                ],
                "Benefits": (
                    "Hydration aur uneven skin appearance "
                    "ko improve karne mein help kar sakte hain."
                ),
                "Precaution": (
                    "Serum optional hai; ek samay par ek naya "
                    "active ingredient introduce karein."
                )
            }
        },

        "Combination": {
            "Cleanser": {
                "Ingredients": [
                    "Glycerin",
                    "Salicylic Acid (optional)"
                ],
                "Benefits": (
                    "Gentle cleanser skin ko clean karta hai. "
                    "Salicylic acid oily areas ke liye useful ho sakta hai."
                ),
                "Look For": (
                    "Gentle cleanser; acid daily use karna zaroori nahi."
                )
            },

            "Moisturizer": {
                "Ingredients": [
                    "Ceramides",
                    "Glycerin",
                    "Hyaluronic Acid"
                ],
                "Benefits": (
                    "Dry areas ko hydrate aur skin barrier ko support "
                    "karne mein help karte hain."
                ),
                "Look For": "Lightweight, non-comedogenic moisturizer"
            },

            "Sunscreen": {
                "Ingredients": [
                    "Zinc Oxide",
                    "Titanium Dioxide",
                    "OR approved organic UV filters"
                ],
                "Benefits": (
                    "UV protection ke liye broad-spectrum SPF 30+ "
                    "choose karein."
                ),
                "Look For": "Broad-spectrum SPF 30+"
            },

            "Optional Serum": {
                "Ingredients": [
                    "Niacinamide (around 2-5%)",
                    "Hyaluronic Acid"
                ],
                "Benefits": (
                    "Niacinamide oiliness ko manage karne mein help "
                    "kar sakta hai; hyaluronic acid hydration deta hai."
                ),
                "Precaution": (
                    "Skin sensitivity ke according product choose karein."
                )
            }
        }
    }

    skin_type = skin_type.strip().title()

    if skin_type not in recommendations:
        return None

    return recommendations[skin_type]


# ==========================================
# Display recommendations
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("           SKINO_AI INGREDIENT GUIDE")
    print("=" * 55)

    skin_type = input(
        "\nEnter skin type (Dry/Oily/Normal/Combination): "
    )

    result = get_recommendations(skin_type)

    if result:
        print(f"\nSkin Type: {skin_type.title()}")

        for category, details in result.items():

            print(f"\n{'-' * 45}")
            print(category.upper())
            print("-" * 45)

            print("Ingredients to look for:")

            for ingredient in details["Ingredients"]:
                print(f"  - {ingredient}")

            print("\nBenefits:")
            print(details["Benefits"])

            print("\nProduct selection:")
            print(details["Look For"])

            if "Precaution" in details:
                print("\nPrecaution:")
                print(details["Precaution"])

        print("\n" + "=" * 55)
        print("MORNING ROUTINE")
        print("Cleanser -> Moisturizer -> Sunscreen")

        print("\nNIGHT ROUTINE")
        print("Cleanser -> Moisturizer")

        print("\nNote:")
        print(
            "Patch-test new products. Stop using products "
            "that cause persistent irritation."
        )

    else:
        print("Invalid skin type. Please try again.")