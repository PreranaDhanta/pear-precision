# Generic, stage-based IPM recommendations for pears.
# Always verify locally and follow labels/PHI/REI and pollinator safety.

RULES = {
  "dormant": [
    "Horticultural oil for overwintering scale/mites on a dry day above freezing.",
    "Copper (delayed-dormant) to reduce fire blight inoculum; avoid phytotoxicity."
  ],
  "green_tip": [
    "If scab risk (wet/cool), begin protective fungicide at green-tip; repeat 7–10 days if wet persists."
  ],
  "pink": [
    "Continue scab protection if rainy; avoid mixing captan with oil within 7 days."
  ],
  "bloom": [
    "Fire blight: apply bactericide per model if temps trigger risk; protect blossoms (rotate actives).",
    "Avoid insecticides harmful to pollinators during bloom."
  ],
  "petal_fall": [
    "Resume insect control where needed and continue scab if conditions favor infection.",
    "Avoid carbaryl near petal fall to reduce fruit drop risk."
  ],
  "first_cover": [
    "Maintain scab/mildew coverage at 10–14 day intervals when wet/humid; rotate FRAC groups."
  ],
  "summer": [
    "Scout regularly; treat only if thresholds exceeded. Manage vigor/irrigation to reduce blight risk."
  ],
  "post_harvest": [
    "Sanitation: remove mummies, prune blight strikes, and clean tools."
  ]
}

def get_recommendations(stage: str):
    stage = stage.lower().strip()
    if stage == "all":
        return [{"stage": k, "guidelines": v} for k,v in RULES.items()]
    return [{"stage": stage, "guidelines": RULES.get(stage, [])}]
