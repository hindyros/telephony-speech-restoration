# Sandbox Action List (Harper Valley Banking)

The sandbox mirrors the 8 caller task types in the [Gridspace Stanford Harper Valley](https://github.com/cricketclub/gridspace-stanford-harper-valley) dataset. Each caller clip is labeled with the task the caller was assigned, and the agent under test must choose the matching action.

| ID | Harper Valley `task_type` | Expected Action | Ground Truth Label |
|----|---------------------------|-----------------|--------------------|
| B01 | `replace card` | Replace a lost or stolen card | `replace_card` |
| B02 | `transfer money` | Transfer money between accounts | `transfer_money` |
| B03 | `check balance` | Report current account balance | `check_balance` |
| B04 | `order checks` | Order a new checkbook | `order_checks` |
| B05 | `pay bill` | Schedule or confirm a bill payment | `pay_bill` |
| B06 | `reset password` | Reset the caller's online banking password | `reset_password` |
| B07 | `schedule appointment` | Book an in-branch appointment | `schedule_appointment` |
| B08 | `get branch hours` | Report branch operating hours | `get_branch_hours` |

## Ground Truth Source

Ground truth comes from the Harper Valley metadata file for each conversation:

```
metadata/<sid>.json -> tasks[0].task_type
```

The label mapping above converts the human-readable `task_type` string into a stable snake_case action label used by the evaluator.

## File Naming

All evaluation WAVs follow the pattern `<action_id>_<n>.wav`, for example `B03_05.wav`. The action ID determines the ground truth; the suffix just disambiguates samples.

## Scoring

- **Correct (1):** Agent returns the exact expected action label
- **Incorrect (0):** Agent returns a different label, asks for clarification, or fails

Correct-action rate per condition = sum of correct scores / total samples.
