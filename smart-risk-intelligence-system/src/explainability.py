def explain_attack(features):
    """
    Simple rule-based explanation for cyber risk
    """

    reasons = []

    if features[0] > 1:
        reasons.append("Long connection duration")

    if features[4] > 1:
        reasons.append("High data transfer")

    if features[5] > 1:
        reasons.append("Unusual traffic pattern")

    if not reasons:
        reasons.append("Normal behavior")

    return reasons