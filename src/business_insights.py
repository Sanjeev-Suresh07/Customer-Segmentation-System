SEGMENT_NAMES = {
    0: "Bronze Regular Customers",
    1: "Gold High Value Customers",
    2: "Silver Regular Customers",
    3: "Gold Premium Customers",
    4: "Silver Inactive Customers",
    5: "Bronze Discount Customers",
    6: "Silver Moderate Customers"
}


MARKETING_STRATEGIES = {
    "Bronze Regular Customers":
        "Use personalized recommendations and loyalty rewards to encourage higher spending.",

    "Gold High Value Customers":
        "Focus on retention, exclusive offers, loyalty benefits, and personalized promotions.",

    "Silver Regular Customers":
        "Encourage repeat purchases through targeted recommendations and moderate promotions.",

    "Gold Premium Customers":
        "Provide premium products, VIP benefits, exclusive offers, and priority rewards.",

    "Silver Inactive Customers":
        "Use re-engagement campaigns, reminders, and limited-time offers to encourage return purchases.",

    "Bronze Discount Customers":
        "Use targeted discounts, bundles, and personalized offers to increase purchase frequency.",

    "Silver Moderate Customers":
        "Encourage repeat purchases using bundles, personalized recommendations, and loyalty incentives."
}


def assign_segment_names(df):
    """Assign customer segment names and marketing strategies."""

    df = df.copy()

    df["Segment"] = df["Cluster"].map(
        SEGMENT_NAMES
    )

    df["Marketing Strategy"] = df["Segment"].map(
        MARKETING_STRATEGIES
    )

    return df