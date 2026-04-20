# Sandbox Action List

Each action has an unambiguous ground-truth outcome. The agent's response is scored as correct (1) or incorrect (0).

Target: ~10 actions. Fill in as the sandbox is built.

| ID | Utterance (example) | Expected Action | Ground Truth Label |
|----|--------------------|-----------------|--------------------|
| A01 | "I'd like to file a return for my order" | `initiate_return` | `initiate_return` |
| A02 | "What's the status of my order?" | `check_order_status` | `check_order_status` |
| A03 | "Can I speak to a human agent?" | `transfer_to_human` | `transfer_to_human` |
| A04 | "I want to cancel my subscription" | `cancel_subscription` | `cancel_subscription` |
| A05 | "Update my shipping address" | `update_address` | `update_address` |
| A06 | "I never received my package" | `report_missing_package` | `report_missing_package` |
| A07 | "Apply my promo code SAVE20" | `apply_promo_code` | `apply_promo_code` |
| A08 | "I want to track my delivery" | `track_delivery` | `track_delivery` |
| A09 | "Refund my last charge" | `initiate_refund` | `initiate_refund` |
| A10 | "Change my password" | `update_password` | `update_password` |

## Scoring

- **Correct (1):** Agent takes the exact expected action
- **Incorrect (0):** Agent takes wrong action, asks for clarification, or fails

Correct-action rate per condition = sum of correct scores / total samples
